from math import sqrt, log
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.ops import roi_align
from einops.einops import rearrange, repeat

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
    
class FinePreprocess(nn.Module):
    def __init__(self, config, d_c_model, cf_res=None, feat_ids=None, feat_dims=None):
        super().__init__()
        self.config = config
        self.cat_c_feat = config['concat_coarse_feat']
        self.cat_c_type = config['concat_coarse_feat_type']
        self.W = self.config['window_size']  # window size of fine-level (cf_res[-1])
        self.ms_feat = config['ms_feat']  # if True, use forward_multi_scale when len(feat_ids)==2
        self.ms_feat_type = config['ms_feat_type']
        
        if self.cat_c_feat:
            if self.cat_c_type == 'nearest':
                _down_proj_layers = [nn.Linear(d_c_model, config['d_model'], bias=True)]
                if config['coarse_layer_norm']:
                    _down_proj_layers = [nn.LayerNorm(d_c_model), *_down_proj_layers]
                self.down_proj = nn.Sequential(*_down_proj_layers)
                
                self.merge_feat = nn.Linear(2*config['d_model'], config['d_model'], bias=True)
            elif self.cat_c_type == 'bilinear':
                _down_proj_layers = [nn.Conv2d(d_c_model, config['d_model'], 3, 1, 1)]
                if config['coarse_layer_norm']:
                    _down_proj_layers = [nn.LayerNorm(d_c_model, self.W, self.W), *_down_proj_layers]
                self.down_proj = nn.Sequential(*_down_proj_layers)

                self.merge_feat = nn.Conv2d(2*config['d_model'], config['d_model'], 3, 1, 1)
           
        # TODO: Refactor net init and change window-size to image scale
        self.coarse_id, self.fine_id = [int(log(r, 2)) for r in cf_res]  # coarse, fine resolutions
        self.feat_ids = feat_ids  # FIXME: This argument is linked to RESFPN, where res2/deit backbone don't have?
        self.feat_dims = feat_dims  # dim of feats returned by backbone
        if self.feat_ids is None or (len(self.feat_ids) == 2 and not self.ms_feat):
            assert self.feat_ids[0] > self.feat_ids[1]
        else:  # multi-scale features
            assert not self.cat_c_feat
            self.WS = [1 if i==self.coarse_id else int(2**(log(self.W-1, 2)-i+self.fine_id))+1 for i in feat_ids]  # multi-scale window sizes
            self.W_MAX = max(self.WS)
            # build networks
            d_hid = config['d_ms_feat']
            if self.ms_feat_type == 'PROJ_MERGE':  # project multi-scale features and merge
                # FIXME: Linear is not suitable here, use multiple 3x3 convs to extract low level features!
                _ms_proj_layers = [[nn.Linear(d, d_hid, bias=True)] for d in self.feat_dims]  # FIXME: bias=False if layernorm
                _ms_proj_layers = [ls if id!=self.coarse_id else [nn.LayerNorm(self.feat_dims[i]), *ls]
                                    for i, (id, ls) in enumerate(zip(feat_ids, _ms_proj_layers))]
                self.ms_proj = nn.ModuleList(nn.Sequential(*ls) for ls in _ms_proj_layers)
                # self.ms_proj = nn.ModuleList([
                #     nn.Sequential(nn.Linear(d, d_hid, bias=False),
                #                   nn.BatchNorm1d(d_hid),
                #                   nn.ReLU(),
                #                   nn.Linear(d_hid, d_hid, bias=True))  # add a final linear projection
                #     for d in self.feat_dims])
                
                ## opt-1: add and merge
                self.ms_merge = nn.Sequential(nn.Linear(d_hid, config['d_model'], bias=True))
                # self.ms_merge = nn.Sequential(nn.Linear(d_hid, config['d_model'], bias=False),
                #                               nn.BatchNorm1d(config['d_model']),
                #                               nn.ReLU(),
                #                               nn.Linear(config['d_model'], config['d_model'], bias=True))

                ## opt-2: cat and merge
                # self.ms_merge = nn.Sequential(nn.Linear(d_hid*len(self.feat_dims), config['d_model'], bias=True))
            elif self.ms_feat_type == 'CAT_CONV':  # extract low-level feature within patches with convolutions
                self.ms_merge = nn.Sequential(OrderedDict([
                    ('conv_norm_act_0', conv_norm_relu(sum(self.feat_dims), d_hid, 3, 1, 1, win_size=self.W_MAX, norm=nn.BatchNorm2d)),
                    ('conv_norm_act_1', conv_norm_relu(d_hid, d_hid, 3, 1, 1, win_size=self.W_MAX, norm=nn.BatchNorm2d)),
                    ('final_proj', nn.Conv2d(d_hid, config['d_model'], 1, bias=True))
                ]))
            else:
                raise ValueError()
            
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            # TODO: use xavier if no relu used
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        
    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data,
                feats0=None, feats1=None):
        data.update({'W': self.W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, getattr(self, 'W_MAX', self.W)**2, self.config['d_model'], device=feat_f0.device)
            feat1 = torch.empty(0, getattr(self, 'W_MAX', self.W)**2, self.config['d_model'], device=feat_f0.device)
            return feat0, feat1
        
        if self.feat_ids is None or (len(self.feat_ids) == 2 and not self.ms_feat):
            return self._forward_single_scale(feat_f0, feat_f1, feat_c0, feat_c1, data)
        else:
            # replace the backbone coarse feat with loftr coarse feat
            feats0 = [f if i != self.coarse_id else feat_c0 for i, f in zip(self.feat_ids, feats0)]
            feats1 = [f if i != self.coarse_id else feat_c1 for i, f in zip(self.feat_ids, feats1)]
            return self._forward_multi_scale(feats0, feats1, data)
    
    def _forward_single_scale(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]
        # if use bilinear upsample, upsample feat_coarse first
        # TODO: No need to convolve on the full feature map?
        if self.cat_c_feat and (self.cat_c_type == 'bilinear'):
            feat_c0 = rearrange(feat_c0, 'n (h w) c -> n c h w', h=data['hw0_c'][0], w=data['hw0_c'][1])
            feat_c1 = rearrange(feat_c1, 'n (h w) c -> n c h w', h=data['hw1_c'][0], w=data['hw1_c'][1])

            feat_c0 = F.upsample(feat_c0, scale_factor=stride, mode='bilinear', align_corners=False)
            feat_c1 = F.upsample(feat_c1, scale_factor=stride, mode='bilinear', align_corners=False)
            feat_c0 = self.down_proj(feat_c0)
            feat_c1 = self.down_proj(feat_c1)
            feat_f0 = self.merge_feat(torch.cat([feat_f0, feat_c0], dim=1))
            feat_f1 = self.merge_feat(torch.cat([feat_f1, feat_c1], dim=1))
            
        # unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        # select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # (n, ww, cf)
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

        # scale = data['hw0_i'][0] // data['hw0_c'][0]
        # feat_f0_unfold = self._extract_local_patches(feat_f0, data['mkpts0_c'], data['m_bids'], 1/scale)
        # feat_f0_unfold = rearrange(feat_f0_unfold, 'n c w h -> n (w h) c')
        # feat_f1_unfold = self._extract_local_patches(feat_f1, data['mkpts1_c'], data['m_bids'], 1/scale)
        # feat_f1_unfold = rearrange(feat_f1_unfold, 'n c w h -> n (w h) c')

        # optional: concat coarse-level loftr feature as context
        # if use nearest coarse ctx
        if self.cat_c_feat and (self.cat_c_type == 'nearest'):
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']], 
                                                   feat_c1[data['b_ids'], data['j_ids']]], 0))  # (2n, c)
            feat_cf_win = self.merge_feat(torch.cat([
                    torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # (2n, ww, cf)
                    repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # (2n, ww, cf)
                ], -1))
            # TODO: add a residual connection?
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)
        return feat_f0_unfold, feat_f1_unfold
    
    def _forward_multi_scale(self, feats0, feats1, data):
        """
        TODO: Speed Optimization.
        Args:
            feats0/feats1 (List[Tensor]): multi-scale features, correspond to self.feat_dims & self.WS
        """
        N = data['b_ids'].shape[0]
        
        def unfold_proj(feat, feat_id, w, l_ids, proj=None):
            if feat.dim() == 4:
                stride = 2 ** (self.coarse_id - feat_id)
                feat_unfold = rearrange(F.unfold(feat, kernel_size=(w, w), stride=stride, padding=w//2),
                                        'n (c ww) l -> n l ww c', ww=w**2)[data['b_ids'], l_ids]  # (n, ww, c)
            else:  # coarse-level loftr feature
                feat_unfold = feat[data['b_ids'], l_ids][:, None]
            feat_proj = rearrange(proj(feat_unfold.flatten(0, 1)), '(n ww) c -> n ww c', n=N) if proj is not None else feat_unfold
            # TODO: bilinaer interpolation?
            feat_resize = F.interpolate(rearrange(feat_proj, 'n (w0 w1) c -> n c w0 w1', w0=w, w1=w),
                                        size=(self.W_MAX, self.W_MAX), mode='nearest', align_corners=None)
            return feat_resize.flatten(-2)
        
        if self.ms_feat_type == 'PROJ_MERGE':
            # NOTE: Use [Resize - Concat - Proj] to sacrifice space for time
            # 1. preprocess features from each scale: unfold -> proj -> resize(max_W)
            feats0 = [unfold_proj(*args, data['i_ids']) for args in zip(feats0, self.feat_ids, self.WS, self.ms_proj)]
            feats1 = [unfold_proj(*args, data['j_ids']) for args in zip(feats1, self.feat_ids, self.WS, self.ms_proj)]
            # 2. add features from each scale -> flatten -> proj
            ## opt-1. add and merge
            feat0 = rearrange(self.ms_merge(rearrange(sum(feats0), 'n c ww -> (n ww) c')), '(n ww) c -> n ww c', n=N)
            feat1 = rearrange(self.ms_merge(rearrange(sum(feats1), 'n c ww -> (n ww) c')), '(n ww) c -> n ww c', n=N)
            
            ## opt-2. cat and merge
            # feat0 = rearrange(self.ms_merge(rearrange(torch.cat(feats0, dim=1), 'n c ww -> (n ww) c')), '(n ww) c -> n ww c', n=N)
            # feat1 = rearrange(self.ms_merge(rearrange(torch.cat(feats1, dim=1), 'n c ww -> (n ww) c')), '(n ww) c -> n ww c', n=N)
        elif self.ms_feat_type == 'CAT_CONV':
            feats0 = [unfold_proj(*args, data['i_ids']) for args in zip(feats0, self.feat_ids, self.WS)]
            feats1 = [unfold_proj(*args, data['j_ids']) for args in zip(feats1, self.feat_ids, self.WS)]
            
            feat0 = rearrange(self.ms_merge(rearrange(torch.cat(feats0, dim=1), 'n c (w0 w1) -> n c w0 w1', w0=self.W_MAX, w1=self.W_MAX)),
                              'n c w0 w1 -> n (w0 w1) c', w0=self.W_MAX, w1=self.W_MAX)
            feat1 = rearrange(self.ms_merge(rearrange(torch.cat(feats1, dim=1), 'n c (w0 w1) -> n c w0 w1', w0=self.W_MAX, w1=self.W_MAX)), 
                              'n c w0 w1 -> n (w0 w1) c', w0=self.W_MAX, w1=self.W_MAX)
        else:
            raise ValueError()
        
        return feat0, feat1

    # NOTE: only test here, remove in future
    def _extract_local_patches(
            self,
            features,  # (N, C, H, W)
            keypoints,  # [L, 2]
            bids,  # [L, 1]
            scale=1
        ):
        bids = bids.unsqueeze(-1) if len(bids.shape) == 1 else bids
        redius = self.W // 2
        boxes = torch.cat([bids, keypoints - redius, keypoints + redius], dim=-1) # L*5
        unfold_features = roi_align(features, boxes, output_size=(self.W, self.W), spatial_scale=scale, aligned=False )
        return unfold_features

class FinePreprocessUnfoldNoneGrid(nn.Module):
    # TODO: move configs to new configs used for fine level matched keypoints correspondence
    def __init__(self, config, d_c_model, cf_res=None, feat_ids=None, feat_dims=None):
        super().__init__()
        self.config = config
        self.cat_c_feat = config['concat_coarse_feat']
        self.cat_c_type = config['concat_coarse_feat_type']
        self.ms_feat = config['ms_feat']  # if True, use forward_multi_scale when len(feat_ids)==2
        self.ms_feat_type = config['ms_feat_type']
        
        if self.cat_c_feat:
            if self.cat_c_type == 'nearest':
                _down_proj_layers = [nn.Linear(d_c_model, config['d_model'], bias=True)]
                if config['coarse_layer_norm']:
                    _down_proj_layers = [nn.LayerNorm(d_c_model), *_down_proj_layers]
                self.down_proj = nn.Sequential(*_down_proj_layers)
                
                self.merge_feat = nn.Linear(2*config['d_model'], config['d_model'], bias=True)
            elif self.cat_c_type == 'bilinear':
                _down_proj_layers = [nn.Conv2d(d_c_model, config['d_model'], 3, 1, 1)]
                if config['coarse_layer_norm']:
                    _down_proj_layers = [nn.LayerNorm(d_c_model, self.W, self.W), *_down_proj_layers]
                self.down_proj = nn.Sequential(*_down_proj_layers)

                self.merge_feat = nn.Conv2d(2*config['d_model'], config['d_model'], 3, 1, 1)
           
        # TODO: Refactor net init and change window-size to image scale
        self.coarse_id, self.fine_id = [int(log(r, 2)) for r in cf_res]  # coarse, fine resolutions
        self.feat_ids = feat_ids  # FIXME: This argument is linked to RESFPN, where res2/deit backbone don't have?
        self.feat_dims = feat_dims  # dim of feats returned by backbone
        if self.feat_ids is None or (len(self.feat_ids) == 2 and not self.ms_feat):
            assert self.feat_ids[0] > self.feat_ids[1]
        else:  # multi-scale features
            assert not self.cat_c_feat
            self.WS = [1 if i==self.coarse_id else int(2**(log(self.W-1, 2)-i+self.fine_id))+1 for i in feat_ids]  # multi-scale window sizes
            self.W_MAX = max(self.WS)
            # build networks
            d_hid = config['d_ms_feat']
            if self.ms_feat_type == 'PROJ_MERGE':  # project multi-scale features and merge
                # FIXME: Linear is not suitable here, use multiple 3x3 convs to extract low level features!
                _ms_proj_layers = [[nn.Linear(d, d_hid, bias=True)] for d in self.feat_dims]  # FIXME: bias=False if layernorm
                _ms_proj_layers = [ls if id!=self.coarse_id else [nn.LayerNorm(self.feat_dims[i]), *ls]
                                    for i, (id, ls) in enumerate(zip(feat_ids, _ms_proj_layers))]
                self.ms_proj = nn.ModuleList(nn.Sequential(*ls) for ls in _ms_proj_layers)
                # self.ms_proj = nn.ModuleList([
                #     nn.Sequential(nn.Linear(d, d_hid, bias=False),
                #                   nn.BatchNorm1d(d_hid),
                #                   nn.ReLU(),
                #                   nn.Linear(d_hid, d_hid, bias=True))  # add a final linear projection
                #     for d in self.feat_dims])
                
                ## opt-1: add and merge
                self.ms_merge = nn.Sequential(nn.Linear(d_hid, config['d_model'], bias=True))
                # self.ms_merge = nn.Sequential(nn.Linear(d_hid, config['d_model'], bias=False),
                #                               nn.BatchNorm1d(config['d_model']),
                #                               nn.ReLU(),
                #                               nn.Linear(config['d_model'], config['d_model'], bias=True))

                ## opt-2: cat and merge
                # self.ms_merge = nn.Sequential(nn.Linear(d_hid*len(self.feat_dims), config['d_model'], bias=True))
            elif self.ms_feat_type == 'CAT_CONV':  # extract low-level feature within patches with convolutions
                self.ms_merge = nn.Sequential(OrderedDict([
                    ('conv_norm_act_0', conv_norm_relu(sum(self.feat_dims), d_hid, 3, 1, 1, win_size=self.W_MAX, norm=nn.BatchNorm2d)),
                    ('conv_norm_act_1', conv_norm_relu(d_hid, d_hid, 3, 1, 1, win_size=self.W_MAX, norm=nn.BatchNorm2d)),
                    ('final_proj', nn.Conv2d(d_hid, config['d_model'], 1, bias=True))
                ]))
            else:
                raise ValueError()
            
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            # TODO: use xavier if no relu used
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        
    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data,
                feats0=None, feats1=None, window_size=5):
        self.W = window_size  # window size of fine-level (cf_res[-1])
        data.update({'W': self.W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, getattr(self, 'W_MAX', self.W)**2, self.config['d_model'], device=feat_f0.device)
            feat1 = torch.empty(0, getattr(self, 'W_MAX', self.W)**2, self.config['d_model'], device=feat_f0.device)
            return feat0, feat1
        
        if self.feat_ids is None or (len(self.feat_ids) == 2 and not self.ms_feat):
            return self._forward_single_scale(feat_f0, feat_f1, feat_c0, feat_c1, data)
        else:
            # replace the backbone coarse feat with loftr coarse feat
            feats0 = [f if i != self.coarse_id else feat_c0 for i, f in zip(self.feat_ids, feats0)]
            feats1 = [f if i != self.coarse_id else feat_c1 for i, f in zip(self.feat_ids, feats1)]
            return self._forward_multi_scale(feats0, feats1, data)
    
    def _forward_single_scale(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]
        # if use bilinear upsample, upsample feat_coarse first
        # TODO: No need to convolve on the full feature map?
        # TODO: remove in fine level matched keypoints crop
        if self.cat_c_feat and (self.cat_c_type == 'bilinear'):
            feat_c0 = rearrange(feat_c0, 'n (h w) c -> n c h w', h=data['hw0_c'][0], w=data['hw0_c'][1])
            feat_c1 = rearrange(feat_c1, 'n (h w) c -> n c h w', h=data['hw1_c'][0], w=data['hw1_c'][1])

            feat_c0 = F.upsample(feat_c0, scale_factor=stride, mode='bilinear', align_corners=False)
            feat_c1 = F.upsample(feat_c1, scale_factor=stride, mode='bilinear', align_corners=False)
            feat_c0 = self.down_proj(feat_c0)
            feat_c1 = self.down_proj(feat_c1)
            feat_f0 = self.merge_feat(torch.cat([feat_f0, feat_c0], dim=1))
            feat_f1 = self.merge_feat(torch.cat([feat_f1, feat_c1], dim=1))
            
        scale = data['hw0_i'][0] // data['hw0_c'][0]
        feat_f0_unfold = self._extract_local_patches(feat_f0, data['mkpts0_f'], data['m_bids'], window_size=self.W, scale=1/scale) # (N, c, w, w)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n c w h -> n (w h) c') # (n, ww, c)
        feat_f1_unfold = self._extract_local_patches(feat_f1, data['mkpts1_f'], data['m_bids'], window_size=self.W, scale=1/scale)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n c w h -> n (w h) c')


        # optional: concat coarse-level loftr feature as context
        # if use nearest coarse ctx
        # TODO: remove in fine level matched keypoints crop
        if self.cat_c_feat and (self.cat_c_type == 'nearest'):
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']], 
                                                   feat_c1[data['b_ids'], data['j_ids']]], 0))  # (2n, c)
            feat_cf_win = self.merge_feat(torch.cat([
                    torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # (2n, ww, cf)
                    repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # (2n, ww, cf)
                ], -1))
            # TODO: add a residual connection?
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)
        return feat_f0_unfold, feat_f1_unfold
    
    def _forward_multi_scale(self, feats0, feats1, data):
        """
        TODO: Speed Optimization.
        Args:
            feats0/feats1 (List[Tensor]): multi-scale features, correspond to self.feat_dims & self.WS
        """
        N = data['b_ids'].shape[0]
        
        def unfold_proj(feat, feat_id, w, l_ids, proj=None):
            if feat.dim() == 4:
                stride = 2 ** (self.coarse_id - feat_id)
                feat_unfold = rearrange(F.unfold(feat, kernel_size=(w, w), stride=stride, padding=w//2),
                                        'n (c ww) l -> n l ww c', ww=w**2)[data['b_ids'], l_ids]  # (n, ww, c)
            else:  # coarse-level loftr feature
                feat_unfold = feat[data['b_ids'], l_ids][:, None]
            feat_proj = rearrange(proj(feat_unfold.flatten(0, 1)), '(n ww) c -> n ww c', n=N) if proj is not None else feat_unfold
            # TODO: bilinaer interpolation?
            feat_resize = F.interpolate(rearrange(feat_proj, 'n (w0 w1) c -> n c w0 w1', w0=w, w1=w),
                                        size=(self.W_MAX, self.W_MAX), mode='nearest', align_corners=None)
            return feat_resize.flatten(-2)
        
        if self.ms_feat_type == 'PROJ_MERGE':
            # NOTE: Use [Resize - Concat - Proj] to sacrifice space for time
            # 1. preprocess features from each scale: unfold -> proj -> resize(max_W)
            feats0 = [unfold_proj(*args, data['i_ids']) for args in zip(feats0, self.feat_ids, self.WS, self.ms_proj)]
            feats1 = [unfold_proj(*args, data['j_ids']) for args in zip(feats1, self.feat_ids, self.WS, self.ms_proj)]
            # 2. add features from each scale -> flatten -> proj
            ## opt-1. add and merge
            feat0 = rearrange(self.ms_merge(rearrange(sum(feats0), 'n c ww -> (n ww) c')), '(n ww) c -> n ww c', n=N)
            feat1 = rearrange(self.ms_merge(rearrange(sum(feats1), 'n c ww -> (n ww) c')), '(n ww) c -> n ww c', n=N)
            
            ## opt-2. cat and merge
            # feat0 = rearrange(self.ms_merge(rearrange(torch.cat(feats0, dim=1), 'n c ww -> (n ww) c')), '(n ww) c -> n ww c', n=N)
            # feat1 = rearrange(self.ms_merge(rearrange(torch.cat(feats1, dim=1), 'n c ww -> (n ww) c')), '(n ww) c -> n ww c', n=N)
        elif self.ms_feat_type == 'CAT_CONV':
            feats0 = [unfold_proj(*args, data['i_ids']) for args in zip(feats0, self.feat_ids, self.WS)]
            feats1 = [unfold_proj(*args, data['j_ids']) for args in zip(feats1, self.feat_ids, self.WS)]

            feat0 = rearrange(self.ms_merge(rearrange(torch.cat(feats0, dim=1), 'n c (w0 w1) -> n c w0 w1', w0=self.W_MAX, w1=self.W_MAX)),
                              'n c w0 w1 -> n (w0 w1) c', w0=self.W_MAX, w1=self.W_MAX)
            feat1 = rearrange(self.ms_merge(rearrange(torch.cat(feats1, dim=1), 'n c (w0 w1) -> n c w0 w1', w0=self.W_MAX, w1=self.W_MAX)), 
                              'n c w0 w1 -> n (w0 w1) c', w0=self.W_MAX, w1=self.W_MAX)
        else:
            raise ValueError()

        return feat0, feat1

    def _extract_local_patches(
            self,
            features,  # (N, C, H, W)
            keypoints,  # [L, 2]
            bids,  # [L, 1]
            window_size=5,
            scale=1
        ):
        '''
        Parameters:
        --------------
        scale :
            keypoints*scale -> feature coordinate
        '''
        bids = bids.unsqueeze(-1) if len(bids.shape)==1 else bids
        redius = self.W // 2
        boxes = torch.cat([bids, keypoints - redius, keypoints + redius], dim=-1) # N*5
        unfold_features = roi_align(features, boxes, output_size=(window_size, window_size), spatial_scale=scale, aligned=False )
        return unfold_features