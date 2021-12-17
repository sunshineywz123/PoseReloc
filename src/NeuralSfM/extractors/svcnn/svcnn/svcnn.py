import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .base_model import BaseModel
from .utils import simple_nms, remove_borders, top_k_keypoints, sample_descriptors, top_k_keypoints_with_descriptor
from .utils import soft_argmax_refinement,quadratic_refinement, refine_with_harris


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2]):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SVCNN(BaseModel):
    default_config = {
        'descriptor_dim': 128,          #pre trained model is 128, can not change
        'nms_radius': 4,
        'refinement_radius': 0,
        'do_quadratic_refinement': 0,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'version': 'v9'
    }
    required_data_keys = ['image']

    def _init(self, config):
        self.config = {**self.default_config, **config}

        self.resnet = ResNet()
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 65, kernel_size=1, stride=1, padding=0)
        )
        if 'v9' in self.config['version']:
            self.descriptor = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
            )
        elif 'little' in self.config['version']:
            self.descriptor = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise NotImplementedError(self.config['version'])

        self.gift = torch.nn.MaxPool1d(kernel_size=len(self.config['scales']))

        path = Path(__file__).parent / 'svcnn_{}.pth'.format(self.config['version'])

        self.load_state_dict(torch.load(str(path))['net'])

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded Svcnn model {}'.format(path))

    def _forward(self, inp, mode='train'):
        scales = self.config['scales']
        b, c, h, w = inp['image'].shape
        results = list()

        for idx, scale in enumerate(scales):
            if idx == 0:
                mask = torch.zeros_like(inp['image'][:, :1])  # [b, 1, h, w]
            n_h, n_w = int(h * scale), int(w * scale)
            assert n_h % 8 == 0 and n_w % 8 == 0, "The width and height of the image must be a multiple of 8 !"
            resized_input = F.interpolate(inp['image'], size=[n_h, n_w], mode='bilinear', align_corners=True)
            single_result = self._forward_single(resized_input, mode, mask=mask)
            if mode == 'train':
                single_result['keypoints'] = (single_result['keypoints'] // scale).long()
            else:
                single_result['keypoints'] = (single_result['keypoints'] / scale)
            results.append(single_result)
            score_map = F.interpolate(single_result['score_map'], size=[h, w], mode='bilinear', align_corners=True)
            mask = mask + score_map

        combined_keys = ['keypoints', 'scores', 'descriptors']
        for key in combined_keys:
            for i in range(1, len(results)):
                if key == 'descriptors':
                    concate_dim = 2
                else:
                    concate_dim = 1
                results[0][key] = torch.cat([results[0][key], results[i][key]], dim=concate_dim)

        multi_scale_results = results[0]
        del multi_scale_results['score_map']

        if self.config['max_keypoints'] >= 0:
            # [N, (max_kpts, 2)], [N, (max_kpts, )] if train
            cutted_results = list(zip(*[
                top_k_keypoints_with_descriptor(k, s, d, self.config['max_keypoints'], h * 8, w * 8, mode)
                for k, s, d in zip(multi_scale_results['keypoints'],
                                   multi_scale_results['scores'],
                                   multi_scale_results['descriptors'])]))
            multi_scale_results['keypoints'], multi_scale_results['scores'], multi_scale_results['descriptors'] = cutted_results

        if mode == 'eval':
            for key, value in multi_scale_results.items():
                multi_scale_results[key] = [x for x in value]
        elif mode == 'train':
            for key, value in multi_scale_results.items():
                multi_scale_results[key] = torch.stack(multi_scale_results[key], 0)

        return multi_scale_results

    def _forward_single(self, image, mode='train', mask=None):
        assert mode in ['train', 'eval', 'dog']
        # semi, ftmap = self.multi_scale_forward(inp)
        ft = self.resnet(image)
        semi = self.detector(ft)
        ftmap = self.descriptor(ft)

        scores = torch.nn.functional.softmax(semi, 1)  # (N, 65, H/8, W/8)
        scores = scores[:, :64]
        b, _, h, w = scores.shape
        # - pixel shuffle
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        # nms
        full_scores = scores
        scores = simple_nms(scores, self.config['nms_radius'])  # (N, H, W)
        score_map = scores.unsqueeze(dim=1)
        if mask is not None:
            resized_mask = F.interpolate(mask, size=[h*8, w*8], mode='bilinear', align_corners=True)[:, 0]
            scores[resized_mask > 0] = 0
        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'], as_tuple=False)
            for s in scores]  # [N, (n_kpts, 2)]
        scores = [s[list(k.t())] for s, k in zip(scores, keypoints)]  # [N, (n_kpts,)] - traverse along batch dim
        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'], h * 8, w * 8, mode)
                for k, s in zip(keypoints, scores)]))  # [N, (max_kpts, 2)], [N, (max_kpts, )] if train

        if self.config['do_quadratic_refinement']:
            keypoints = quadratic_refinement(keypoints, full_scores)
        elif self.config['refinement_radius'] > 0:
            keypoints = soft_argmax_refinement(
                keypoints, full_scores, self.config['refinement_radius'])
        elif self.config['harris_radius'] > 0:
            keypoints = refine_with_harris(keypoints, image, self.config['harris_radius'])
        # Compute the dense descriptors
        descriptors = torch.nn.functional.normalize(ftmap, p=2, dim=1)  # (N, D, H/8, W/8)

        keypoints = torch.stack(keypoints, 0)  # (N, max_kpts, 2) - don't cast to float here
        scores = torch.stack(scores, 0)  # (N, max_kpts, )
        # Convert (h, w) to (x, y)
        keypoints = torch.flip(keypoints, [2])
        # Extract descriptors
        descriptors = sample_descriptors(keypoints, descriptors, 8)  # (N, D, max_kpts)
        if mode == 'eval':
            keypoints = keypoints.float()
        return {
            'keypoints': keypoints,  # [N, (n_kpts, 2)] - (x, y) / (N, max_kpts, 2)
            'scores': scores,  # [N, (n_kpts,)] - scores don't sum up to 1.0 / (N, max_kpts, )
            'descriptors': descriptors,  # [N, (D, n_kpts)] / (N, D, max_kpts)
            'score_map': score_map,  # (N, H, W)
        }

    def _abandon_multi_scale_forward(self, inp):
        scales = self.config['scales']
        if len(scales) <= 1:
            ft = self.resnet(inp['image'])
            semi = self.detector(ft)
            ftmap = self.descriptor(ft)
        else:
            b, c, h, w = inp['image'].shape
            assert h % 8 == 0 and w % 8 == 0, "The width and height of the image must be a multiple of 8 !"
            ftmaps = list()
            for idx, scale in enumerate(scales):
                n_h, n_w = int(h*scale), int(w*scale)
                resized_input = F.interpolate(inp['image'], size=[n_h, n_w], mode='bilinear', align_corners=True)
                ft = self.resnet(resized_input)
                if idx == 0:
                    semi = F.interpolate(self.detector(ft), size=[h//8, w//8], mode='bilinear', align_corners=True)
                ftmap = F.interpolate(self.descriptor(ft), size=[h//8, w//8], mode='bilinear', align_corners=True)
                ftmaps.append(ftmap)
            ftmaps_torch = torch.stack(ftmaps, dim=-1)
            b, c, h, w, l = ftmaps_torch.shape
            ftmaps_torch_n = ftmaps_torch.view(b, c * h * w, l)
            ftmap = self.gift(ftmaps_torch_n).view(b, c, h, w)
        return semi, ftmap

