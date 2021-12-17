import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from pathlib import Path

from .utils.exceptions import EmptyTensorError
from .utils.utils import interpolate_dense_features, upscale_positions


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, finetune_feature_extraction=False,):
        super(DenseFeatureExtractionModule, self).__init__()

        model = models.vgg16()
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv4_3_idx = vgg16_layers.index('conv4_3')

        self.model = nn.Sequential(
            *list(model.features.children())[: conv4_3_idx + 1]
        )

        self.num_channels = 512

        # Fix forward parameters
        for param in self.model.parameters():
            param.requires_grad = False
        if finetune_feature_extraction:
            # Unlock conv4_3
            for param in list(self.model.parameters())[-2 :]:
                param.requires_grad = True

    def forward(self, batch):
        output = self.model(batch)
        return output


class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)

        return score


class D2Net(nn.Module):
    def __init__(self, config):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            finetune_feature_extraction=True,
        )

        self.config=config
        self.detection = SoftDetectionModule()
        self.scales=config['scales']
        self.max_keypoints=config['max_keypoints']

        path = Path(__file__).parent / 'models'/'d2_{}.pth'.format(self.config['version'])
        self.load_state_dict(torch.load(path)['model'])
    '''
    def forward(self, batch):
        b = batch['image1'].size(0)

        dense_features = self.dense_feature_extraction(
            torch.cat([batch['image1'], batch['image2']], dim=0)
        )

        scores = self.detection(dense_features)

        dense_features1 = dense_features[: b, :, :, :]
        dense_features2 = dense_features[b :, :, :, :]

        scores1 = scores[: b, :, :]
        scores2 = scores[b :, :, :]

        return {
            'dense_features1': dense_features1,
            'scores1': scores1,
            'dense_features2': dense_features2,
            'scores2': scores2
        }
    '''


    def forward(self,data,mode):
        b, c, h_init, w_init = data['image'].size()
        device = data['image'].device
        assert(b == 1)
        images=data['image']
        images=images.expand(-1,3,-1,-1) if c==1 else images

        all_keypoints = torch.zeros([3, 0])
        all_descriptors = torch.zeros([
            self.dense_feature_extraction.num_channels, 0
        ])
        all_scores = torch.zeros(0)

        previous_dense_features = None
        banned = None
        for idx, scale in enumerate(self.scales):
            current_image = F.interpolate(
                images, scale_factor=scale,
                mode='bilinear', align_corners=True
            )
            _, _, h_level, w_level = current_image.size()

            dense_features = self.dense_feature_extraction(current_image)
            del current_image

            _, _, h, w = dense_features.size()

            # Sum the feature maps.
            if previous_dense_features is not None:
                dense_features += F.interpolate(
                    previous_dense_features, size=[h, w],
                    mode='bilinear', align_corners=True
                )
                del previous_dense_features

            # Recover detections.
            detections = self.detection(dense_features)
            if banned is not None:
                banned = F.interpolate(banned.float(), size=[h, w]).bool()
                detections = torch.min(detections, ~banned)
                banned = torch.max(
                    torch.max(detections, dim=1)[0].unsqueeze(1), banned
                )
            else:
                banned = torch.max(detections, dim=1)[0].unsqueeze(1)
            fmap_pos = torch.nonzero(detections[0]).t()
            del detections

            # Recover displacements.
            displacements = self.localization(dense_features)[0]
            displacements_i = displacements[
                0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
            ]
            displacements_j = displacements[
                1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
            ]
            del displacements

            mask = torch.min(
                torch.abs(displacements_i) < 0.5,
                torch.abs(displacements_j) < 0.5
            )
            fmap_pos = fmap_pos[:, mask]
            valid_displacements = torch.stack([
                displacements_i[mask],
                displacements_j[mask]
            ], dim=0)
            del mask, displacements_i, displacements_j

            fmap_keypoints = fmap_pos[1 :, :].float() + valid_displacements
            del valid_displacements

            try:
                raw_descriptors, _, ids = interpolate_dense_features(
                    fmap_keypoints.to(device),
                    dense_features[0]
                )
            except EmptyTensorError:
                continue
            fmap_pos = fmap_pos[:, ids]
            fmap_keypoints = fmap_keypoints[:, ids]
            del ids

            keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
            del fmap_keypoints

            descriptors = F.normalize(raw_descriptors, dim=0)
            del raw_descriptors

            keypoints[0, :] *= h_init / h_level
            keypoints[1, :] *= w_init / w_level

            fmap_pos = fmap_pos
            keypoints = keypoints

            keypoints = torch.cat([
                keypoints,
                torch.ones([1, keypoints.size(1)]) * 1 / scale,
            ], dim=0)

            scores = dense_features[
                0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
            ] / (idx + 1)
            del fmap_pos

            all_keypoints = torch.cat([all_keypoints, keypoints], dim=1)
            all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
            all_scores = torch.cat([all_scores, scores], dim=0)
            del keypoints, descriptors

            previous_dense_features = dense_features
            del dense_features
        del previous_dense_features, banned

        keypoints = all_keypoints.t()
        del all_keypoints
        scores = all_scores
        del all_scores
        descriptors = all_descriptors.t()
        del all_descriptors
        return {
            "keypoints": keypoints,
            "scores": scores,
            "descriptors":descriptors
        }
