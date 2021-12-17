import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import block_type, conv1x1


class ResCoarse(nn.Module):
    default_cfg = {
        'block_type': 'BasicBlock',
        'initial_dim': 64,
        'block_dims': [64, 96, 128]
    }

    def __init__(self, config):
        super().__init__()
        config = {**self.default_cfg, **config}

        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer3_proj = conv1x1(block_dims[2], block_dims[2])
        if len(block_dims) == 4:
            del self.layer3_proj
            self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16
            self.layer4_proj = conv1x1(block_dims[3], block_dims[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        if hasattr(self, 'layer4'):
            x4 = self.layer4(x3)  # 1/16
            return self.layer4_proj(x4)
        else:
            return self.layer3_proj(x3)


class ResFine(nn.Module):
    default_cfg = {
        'block_type': 'BasicBlock',
        'initial_dim': 128,
        'block_dims': [128]
    }

    def __init__(self, config):
        super().__init__()
        config = {**self.default_cfg, **config}

        block = block_type[config['block_type']]
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        if len(block_dims) == 2:
            self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layer3 = block(dim, dim, stride=1)
        layers = (layer1, layer2, layer3)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        if hasattr(self, 'layer2'):
            return self.layer2(x1)  # 1/4
        else:
            return x1


class DoubleRes18(nn.Module):
    """Res18-c4 for coarse-level + Res18-c2 for fine-level"""

    default_cfg = {
        'coarse': {
            'block_type': 'BasicBlock',
            'initial_dim': 64,
            'block_dims': [64, 96, 128]
        },
        'fine': {
            'block_type': 'BasicBlock',
            'initial_dim': 128,
            'block_dims': [128]
        }
    }

    def __init__(self, config, resolution):
        super().__init__()
        self.config = {**self.default_cfg, **config}
        assert (resolution[0] == 8) and (resolution[1] == 2), "only supports resolution from 8->2"
        self.coarse_pretrained = config['coarse_pretrained']

        self.resnet_coarse = ResCoarse(self.config['coarse'])
        self.resnet_fine = ResFine(self.config['fine'])

        # if None, then training from scratch; elif string, then load, detach and run in eval.
        if self.coarse_pretrained is not None:
            ckpt = torch.load(self.coarse_pretrained, 'cpu')['state_dict']
            for k in list(ckpt.keys()):
                if 'resnet_coarse' in k:
                    newk = k[k.find('resnet_coarse')+len('resnet_coarse')+1:]
                    ckpt[newk] = ckpt[k]
                ckpt.pop(k)
            self.resnet_coarse.load_state_dict(ckpt)

            for param in self.resnet_coarse.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.coarse_pretrained is not None:
            self.resnet_coarse.eval()
        c_out = self.resnet_coarse(x)
        f_out = self.resnet_fine(x)

        return [c_out, f_out]


class DoubleRes18_16_4(nn.Module):
    """Res18-c4 for coarse-level + Res18-c2 for fine-level"""

    default_cfg = {
        'coarse': {
            'block_type': 'BasicBlock',
            'initial_dim': 64,
            'block_dims': [64, 96, 128, 256]
        },
        'fine': {
            'block_type': 'BasicBlock',
            'initial_dim': 64,
            'block_dims': [64, 128]
        }
    }

    def __init__(self, config, resolution):
        super().__init__()
        self.config = {**self.default_cfg, **config}
        assert (resolution[0] == 16) and (resolution[1] == 4), "only supports resolution from 16->4"
        self.coarse_pretrained = config['coarse_pretrained']

        self.resnet_coarse = ResCoarse(self.config['coarse'])
        self.resnet_fine = ResFine(self.config['fine'])

        # if None, then training from scratch; elif string, then load, detach and run in eval.
        if self.coarse_pretrained is not None:
            ckpt = torch.load(self.coarse_pretrained, 'cpu')['state_dict']
            for k in list(ckpt.keys()):
                if 'resnet_coarse' in k:
                    newk = k[k.find('resnet_coarse')+len('resnet_coarse')+1:]
                    ckpt[newk] = ckpt[k]
                ckpt.pop(k)
            self.resnet_coarse.load_state_dict(ckpt)

            for param in self.resnet_coarse.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.coarse_pretrained is not None:
            self.resnet_coarse.eval()
        c_out = self.resnet_coarse(x)
        f_out = self.resnet_fine(x)

        return [c_out, f_out]
