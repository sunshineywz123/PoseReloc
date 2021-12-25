import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models import create_model

from . import resnet
from . import deit
from src.utils import weight_init


class HybridBackbone(nn.Module):
    """DeiT for coarse-level + ResNet18 for fine-level"""
    default_cfg = {
        'deit_dim': 192,
        'resnet_dim': [64, 128, 256, 512]
    }
    
    def __init__(self, config, resolution):
        super().__init__()
        if resolution[0] != 16:
            assert config['patch_stride'] == resolution[0]
        if not config['pretrained']:
            raise NotImplementedError(f"Only pretrained model arch supported yet.")
        
        self.deit = create_model('deit_tiny_patch16',
                                 pretrained=config['pretrained'],
                                 img_size=config['img_size'],
                                 in_chans=config['in_chans'],
                                 stride=config['patch_stride'],
                                 depth=config['depth'],
                                 attn_type=config['attn_type'],
                                 attn_cfg=config['attn_cfg'])
        self.resnet = create_model(f'resnet18_c{int(math.log(resolution[1],2))+1}',
                                   prettrained=config['pretrained'],
                                   in_chans=config['in_chans'])
        self.coarse_proj = nn.Conv2d(self.default_cfg['deit_dim'],
                                     config['output_dims'][0], 1, 1, 0)
        self.fine_proj = nn.Conv2d(self.default_cfg['resnet_dim'][int(math.log(resolution[1],2))-1],
                                   config['output_dims'][1], 1, 1, 0)
        
        for m in [self.coarse_proj, self.fine_proj]:
            weight_init.c2_msra_fill(m)
    
    def forward(self, x):
        c_feat = self.deit(x)
        c_out = self.coarse_proj(c_feat)
        
        f_feat = self.resnet(x)
        f_out = self.fine_proj(f_feat)
        
        return [c_out, f_out]