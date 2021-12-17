from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from einops import rearrange


def build_rejector(config):
    enable = config['rejector']['enable']
    d_model = config['d_model']
    win_size = config['window_size']
    if enable:
        return FineRejector(d_model, win_size)
    else:
        return PlaceHolder(d_model, win_size)


def build_norm(norm, nc, l0=None, l1=None):
    if issubclass(norm, nn.LayerNorm):
        n_shape = list(filter(lambda x: x is not None, [nc, l0, l1]))
        return partial(norm, n_shape, elementwise_affine=True)
    elif issubclass(norm, _BatchNorm):
        return partial(norm, nc, affine=True)
    else:
        raise NotImplementedError()


def conv_norm_relu(nc_in, nc_out, kernel_size, stride, 
                   padding=0, win_size=5, norm=nn.LayerNorm, relu=nn.ReLU):
    bias = False if issubclass(norm, _BatchNorm) else True
    norm = build_norm(norm, nc_out, win_size, win_size)
    return nn.Sequential(
        nn.Conv2d(nc_in, nc_out, kernel_size, stride, padding, bias=bias),
        norm(),
        relu()
    )


def fc_norm_relu(nc_in, nc_out, norm=nn.LayerNorm, relu=nn.ReLU):
    bias = False if issubclass(norm, _BatchNorm) else True
    norm = build_norm(norm, nc_out)
    return nn.Sequential(
        nn.Linear(nc_in, nc_out, bias=bias), norm(), relu())


class FineRejector(nn.Module):
    def __init__(self, d_model, win_size):
        super().__init__()
        self.w = win_size
        mp_filter = win_size // 2 if win_size % 2 == 0 else win_size // 2 + 1
        d_model = d_model * 2
        self.spatial_merge = nn.Sequential(OrderedDict([
            ('conv_norm_act_0', conv_norm_relu(d_model, d_model, 3, 2, 1, win_size=win_size, norm=nn.BatchNorm2d)),
            ('conv_norm_act_1', conv_norm_relu(d_model, d_model, 3, 1, 1, win_size=mp_filter, norm=nn.BatchNorm2d)),
            ('max_pool', nn.MaxPool2d(mp_filter, 1))
        ]))
        self.rejector = nn.Sequential(OrderedDict([
            ('fc_bn_act_0', fc_norm_relu(d_model, d_model, norm=nn.BatchNorm1d)),
            ('fc_bn_act_1', fc_norm_relu(d_model, d_model, norm=nn.BatchNorm1d)),
            ('bi_cls', nn.Linear(d_model, 1, bias=True))
        ]))

    def forward(self, feat0, feat1, data):
        """
        Args:
            feat0 (torch.Tensor): [N, WW, C]
            feat1 (torch.Tensor): [N, WW, C]
            data (dict)
        Update:
            data (dict):{
                'mconf_f' (torch.Tensor): [M, 1]
                }
        """
        if feat0.shape[0] == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            data.update({'conf_f': torch.empty(0, 1, device=feat0.device),
                         'mconf_f': torch.empty(0, device=feat0.device)})
            return
        
        feat0, feat1 = map(lambda x: rearrange(x, 'n (w0 w1) c -> n c w0 w1', w0=self.w, w1=self.w), [feat0, feat1])
        feat = torch.cat([feat0, feat1], dim=1)
        feat = self.spatial_merge(feat).squeeze()
        assert feat.dim() == 2  # TODO: Remove
        logits = self.rejector(feat)
        probs = torch.sigmoid(logits)
        data.update({'conf_f': probs,
                     'mconf_f': probs[~data['gt_mask']][..., 0]})


class PlaceHolder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        args[-1].update({'mconf_f': None})




