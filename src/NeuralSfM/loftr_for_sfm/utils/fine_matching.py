from einops.einops import rearrange
from loguru import logger
from functools import partial
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from .supervision import compute_supervision_fine
from .layers import Lambda
from ..backbone import _get_win_rel_scale


# FIXME: Aggregate all layer related codes.
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
        relu())
    

def fc_norm_relu(nc_in, nc_out, norm=nn.LayerNorm, relu=nn.ReLU):
    bias = False if issubclass(norm, _BatchNorm) else True
    norm = build_norm(norm, nc_out)
    return nn.Sequential(
        nn.Linear(nc_in, nc_out, bias=bias), norm(), relu())
    
    
def build_regressor(cfg, W):
    if cfg['type'] == 'correlation':
        regressor = nn.Sequential(OrderedDict([
            ('fc_bn_act_0', fc_norm_relu(W*W, cfg['d'], norm=nn.BatchNorm1d)),
            ('fc_bn_act_1', fc_norm_relu(cfg['d'], cfg['d'], norm=nn.BatchNorm1d)),
            ('coord_reg', nn.Sequential(
                                nn.Linear(cfg['d'], 2, bias=True),
                                nn.Sigmoid(),
                                Lambda(lambda x: 2 * x - 1)))
        ]))
    elif cfg['type'] == 'diff':
        regressor = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('spatial_merge', nn.Sequential(OrderedDict([
                    ('conv_norm_act_0', conv_norm_relu(cfg['d'], 2*cfg['d'], 3, 2, 1, win_size=W, norm=nn.BatchNorm2d)),
                    ('conv_norm_act_1', conv_norm_relu(2*cfg['d'], 2*cfg['d'], 3, 1, 1, win_size=W//2, norm=nn.BatchNorm2d)),
                    ('max_pool', nn.MaxPool2d(W//2, 1))  # TODO: The MaxPool seems irrational? AvgPool? Pos.Enc?
                ])))])),
            nn.Sequential(OrderedDict([
                ('coord_reg', nn.Sequential(OrderedDict([
                    ('fc_bn_act_0', fc_norm_relu(2*cfg['d'], cfg['d'], norm=nn.BatchNorm1d)),
                    ('fc_bn_act_1', fc_norm_relu(cfg['d'], cfg['d'], norm=nn.BatchNorm1d)),
                    ('reg_head', nn.Sequential(nn.Linear(cfg['d'], 2, bias=True),
                                               nn.Sigmoid(),
                                               Lambda(lambda x: 2 * x - 1)))
                ])))]))
            ])
    else:
        raise ValueError()
        
    return regressor

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm
    NOTE: use a separate class for d2d (sprase/dense flow) ?
    """
    def __init__(self, config, _full_cfg=None):
        super().__init__()
        self.config = config
        self.fine_detector = config['detector']
        self._type = config['s2d']['type']
        
        if self._type == 'regress':
            self.regressor = build_regressor(config['s2d']['regress'],
                                             config['window_size'] * _get_win_rel_scale(_full_cfg['LOFTR_BACKBONE']) - 1)
        
        # for gt visualization
        self._full_cfg = _full_cfg
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            # TODO: use xavier for the final Linear reg-head.
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            logger.warning('No matches found in coarse-level.')
            _out_dim = 3 if self._type == 'heatmap' else 2
            data.update({
                'expec_f': torch.empty(0, _out_dim, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        feat_f0_picked = self.select_left_point(feat_f0, data)
        
        coords_normed = self.predict_s2d(feat_f0_picked, feat_f1, data)
        # compute absolute kpt coords
        self.build_mkpts(coords_normed, data)
        
    def select_left_point(self, feat_f0, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        
        if self.fine_detector == 'OnGrid':
            feat_f0_picked = feat_f0[:, WW//2, :]
        elif self.fine_detector in ['SuperPoint', 'SuperPointEC', 'SIFT'] or 'and grid' in self.fine_detector:
            grid = data['i_associated_kpts_local'][:, None, None, :] / (W // 2 * scale)  # normalized offset
            assert (grid < -1).sum() == 0 and (grid > 1).sum() == 0, f'Fine-level Window size is not big enough: w={W}'
            feat_f0 = rearrange(feat_f0, 'm (w1 w2) c -> m c w1 w2', w1=W, w2=W)
            feat_f0_picked = F.grid_sample(feat_f0, grid, align_corners=True)[:, :, 0, 0]
        else:
            raise NotImplementedError
        return feat_f0_picked
        
    def predict_s2d(self, feat_f0_picked, feat_f1, data):
        # compute normalized coords ([-1, 1]) of right patches
        if self._type == 'heatmap':
            coords_normed = self._s2d_heatmap(feat_f0_picked, feat_f1, data)
        elif self._type == 'regress':
            coords_normed = self._s2d_regress(feat_f0_picked, feat_f1, data)
        else:
            raise NotImplementedError()
        return coords_normed
        
    def _s2d_heatmap(self, feat_f0_picked, feat_f1, data):
        W, WW, C = self.W, self.WW, self.C
        
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5  # FIXME: 1. / C ?
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})
        return coords_normalized
        
    def _s2d_regress(self, feat_f0_picked, feat_f1, data):
        W, WW, C = self.W, self.WW, self.C
        regress_type = self.config['s2d']['regress']['type']
        norm_method = self.config['s2d']['regress']['norm']
            
        def norm_sim_s2d(sim):
            if norm_method == 'feat_dim':
                return sim / C
            elif norm_method == 'l2':
                return F.normalize(sim, p=2, dim=1)
            else:
                raise NotImplementedError()
        
        if regress_type == 'correlation':
            # regress coordinates from correlation results
            sim_s2d = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
            sim_s2d = norm_sim_s2d(sim_s2d)
            
            coords_normed = self.regressor(sim_s2d)
        elif regress_type == 'diff':
            diff_s2d = feat_f0_picked[:, None] - feat_f1  # (M, WW, C)
            # diff_s2d = torch.abs(diff_s2d)
            # TODO: Normalization? (BN, LN, l2, scaling, ...)
            diff_merged = self.regressor[0](rearrange(diff_s2d, 'm (w0 w1) c -> m c w0 w1', w0=W, w1=W))[:, :, 0, 0]  # (M, C)
            coords_normed = self.regressor[1](diff_merged)
        else:
            raise ValueError()
            
        data.update({'expec_f': coords_normed})
        return coords_normed
        
    @torch.no_grad() 
    def build_mkpts(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        
        # mkpts0_f
        if self.fine_detector == 'OnGrid':
            mkpts0_f = data['mkpts0_c']
        elif self.fine_detector in ['SuperPoint', 'SuperPointEC', 'SIFT'] or 'and grid' in self.fine_detector:
            local_scale0 = 1 * data['scale0'][data['b_ids']][:, [1, 0]] if 'scale0' in data else 1
            mkpts0_f = data['mkpts0_c'] + (data['i_associated_kpts_local'] * local_scale0)[:len(data['mconf'])]
        else:
            raise NotImplementedError

        # mkpts1_f
        scale1 = scale * data['scale1'][data['b_ids']][:, [1, 0]] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mkpts0_c'])]

        if self.config['_gt']:  # data['expec_f_gt'] > 1 is possible, which exceeds window but allowed fot gt vis.
            compute_supervision_fine(data, self._full_cfg)
            expec_f_gt = data['expec_f_gt']
            if self.config['_gt_noise'] != 0:
                assert self.config['_gt_noise'] > 0 and self.config['_gt_noise'] <= 1
                noise = (torch.rand_like(expec_f_gt) * 2 - 1) * 2 * self.config['_gt_noise']
                expec_f_gt = expec_f_gt + noise
            if self.config['_gt_trunc']:
                expec_f_gt = expec_f_gt.clamp(-1., 1.)
            mkpts1_f = data['mkpts1_c'] + (expec_f_gt * (W // 2) * scale1)[:len(data['mconf'])]

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })
