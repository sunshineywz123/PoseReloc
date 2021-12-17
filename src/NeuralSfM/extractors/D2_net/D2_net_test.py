import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .utils.exceptions import EmptyTensorError
from .utils.utils import interpolate_dense_features, upscale_positions


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True,):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1),
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )
        self.num_channels = 512

        self.use_relu = use_relu

    def forward(self, batch):
        output = self.model(batch)
        if self.use_relu:
            output = F.relu(output)
        return output


class D2Net(nn.Module):
    def __init__(self,config):
        super(D2Net, self).__init__()
        self.config=config
        self.scales=config['scales']
        self.max_keypoints=config['max_keypoints']

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            use_relu=True
        )

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()

        path = Path(__file__).parent / 'models'/'d2_{}.pth'.format(self.config['version'])
        self.load_state_dict(torch.load(path,map_location='cpu')['model'])

    def forward(self,data,mode):
        b, c, h_init, w_init = data['image'].size()
        device = data['image'].device
        assert(b == 1),"only allow for batch size=1 in D2_net!"
        images=data['image']
        images=images.expand(-1,3,-1,-1) if c==1 else images

        all_keypoints = torch.zeros([3, 0],device=device)
        all_descriptors = torch.zeros([
            self.dense_feature_extraction.num_channels, 0
        ],device=device)
        all_scores = torch.zeros(0,device=device)

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

            keypoints[0, :] *= h_init / h_level  #y
            keypoints[1, :] *= w_init / w_level  #x

            fmap_pos = fmap_pos
            keypoints = keypoints

            keypoints = torch.cat([
                keypoints,
                torch.ones([1, keypoints.size(1)],device=device) * 1 / scale,
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
        descriptors = all_descriptors
        del all_descriptors

        #get top k!
        n=self.max_keypoints+1

        #only for mnn
        n=min(n,scores.shape[0])
        if scores.shape[0]<self.max_keypoints+1:
            print(f"warning:not enough points,only have{scores.shape[0]},condition only allow for mnn")
            self.max_keypoints=n-1

        #constrains for spg
        #assert scores.shape[0] >= self.max_keypoints,f"not enough points,only have{scores.shape[0]}"
        minus_threshold, _indices = torch.kthvalue(-scores, n)
        mask = scores > -minus_threshold 
                
        if mask.float().sum() != self.max_keypoints:
            mask_equal = scores == -minus_threshold
            assert mask_equal.float().sum()!=1,"num of threshold is 1"
            diff=self.max_keypoints-mask.float().sum()
            assert mask_equal.float().sum()>=diff,"num of threhold smaller than diff"
            for i in range(mask_equal.numel()):
                if mask_equal[i]==True:
                    if diff!=0:
                        diff-=1
                    else:
                        mask_equal[i]=False
            mask=mask | mask_equal
            assert mask.float().sum() == self.max_keypoints,"still not equal"
        return {
            "keypoints": keypoints[:,:2][mask][:,[1,0]].unsqueeze(0).long(),
            "scores": scores[mask].unsqueeze(0),
            "descriptors":descriptors[:,mask].unsqueeze(0)
        }


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)
