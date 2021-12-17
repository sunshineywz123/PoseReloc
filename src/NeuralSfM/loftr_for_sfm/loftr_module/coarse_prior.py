from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from einops import rearrange


def build_coarse_prior(config):
    if config['prior']['enable']:
        return CoarsePrior(config['d_model'],
                           detach_input=config['prior']['detach'])
    else:
        return PlaceHolder()


def build_norm(norm, nc, l0=None, l1=None):
    if issubclass(norm, nn.LayerNorm):
        n_shape = list(filter(lambda x: x is not None, [nc, l0, l1]))
        return partial(norm, n_shape, elementwise_affine=True)
    elif issubclass(norm, _BatchNorm):
        return partial(norm, nc, affine=True)
    else:
        raise NotImplementedError()


def conv_norm_relu(nc_in, nc_out, kernel_size, stride, 
                   padding=0, l0=60, l1=80, norm=nn.LayerNorm, relu=nn.ReLU):
    bias = False if issubclass(norm, _BatchNorm) else True
    norm = build_norm(norm, nc_out, l0, l1)
    return nn.Sequential(
        nn.Conv2d(nc_in, nc_out, kernel_size, stride, padding, bias=bias),
        norm(),
        relu()
    )


class CoarsePrior(nn.Module):
    def __init__(self, d_model, detach_input=True):
        super().__init__()
        self.detach = detach_input
        self.layer = nn.Sequential(OrderedDict([
            ('conv3x3_norm_act_0', conv_norm_relu(d_model, d_model, 3, 1, 1, norm=nn.BatchNorm2d, relu=nn.ReLU)),
            ('conv3x3_norm_act_1', conv_norm_relu(d_model, d_model, 3, 1, 1, norm=nn.BatchNorm2d, relu=nn.ReLU)),
            ('conv1x1_norm_act_0', conv_norm_relu(d_model, d_model, 1, 1, 0, norm=nn.BatchNorm2d, relu=nn.ReLU)),
            ('prior_cls', nn.Conv2d(d_model, 1, 1, 1, bias=True))
        ]))

    def forward(self, feat0, feat1, data):
        """
        Args:
            feat0 (torch.Tensor): [N, HW0, C]
            feat1 (torch.Tensor): [N, HW1, C]
            data (dict)
        Update:
            data (dict):{
                'prior0' (torch.Tensor): [N, HW0, 1]
                'prior1' (torch.Tensor): [N, HW1, 1]
                }
        """
        h0, w0 = data['hw0_c']
        h1, w1 = data['hw1_c']
        for i, feat, h, w in [[0, feat0, h0, w0], [1, feat1, h1, w1]]:
            feat = rearrange(feat0, 'n (h w) c -> n c h w', h=h, w=w)
            if self.detach:
                feat = feat.detach()
            logits = self.layer(feat)
            probs = torch.sigmoid(logits)
            # TODO: Set prior in padded region to zero
            data.update({f'prior{i}': rearrange(probs, 'n c h w -> n (h w) c')})


class PlaceHolder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        for i in [0, 1]:
            args[-1].update({f'prior{i}': None})
