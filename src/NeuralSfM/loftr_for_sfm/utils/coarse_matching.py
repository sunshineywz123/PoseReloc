from loguru import logger
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda import ComplexFloatStorage, amp
from einops.einops import rearrange
from src.utils.profiler import PassThroughProfiler

from .optimal_transport import OptimalTransport


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:0] = v
    m[:, :, -b:0] = v
    m[:, :, :, -b:0] = v
    m[:, :, :, :, -b:0] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    # TODO: Vectorization
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0-bd:] = v
        m[b_idx, :, w0-bd:] = v
        m[b_idx, :, :, h1-bd:] = v
        m[b_idx, :, :, :, w1-bd:] = v


def calc_max_candidates(p_m0, p_m1):
    """Calculate the max candidates of all pairs within a batch"""
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def build_feat_normalizer(method, **kwargs):
    if method == 'sqrt_feat_dim':
        return lambda feat: feat / feat.shape[-1]**.5
    elif method == 'none' or method is None:
        return lambda feat: feat
    elif method == 'temparature':
        return lambda feat: feat / kwargs['temparature']
    else:
        raise ValueError


class CoarseMatching(nn.Module):
    """
    TODO: limit max(#coarse-matches) to avoid OOM during fine-level inference.
    """
    def __init__(self,
                 config,
                 fine_detector='OnGrid',
                 guided_matching=False,  # coarse-level only guided matching
                 profiler=None):
        super().__init__()
        self.config = config
        self.fine_detector = fine_detector
        self.guided_matching = guided_matching
        self.feat_normalizer = build_feat_normalizer(config['feat_norm_method'])

        self.type = config['type']
        if self.type == 'sinkhorn':
            # Use Sinkhorn algorithm as default
            self.optimal_transport = OptimalTransport(config['skh'])
            self.skh_prefilter = config['skh']['prefilter']
            self.skh_enable_fp16 = config['skh']['fp16']
        elif self.type == 'dual-softmax':
            self.temperature = config['dual_softmax']['temperature']
        else:
            raise NotImplementedError()

        # from conf_matrix to prediction
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']
        
        self._n_rand_samples = config['_n_rand_samples']
        
        self.profiler = profiler or PassThroughProfiler
        #FIXME: move this para to cfg
        self.save_coarse_all_matches= config['save_coarse_all_matches']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(self.feat_normalizer, [feat_c0, feat_c1])
        
        # TODO: Refactor needed.
        if self.type == 'dual-softmax':
            # dual-softmax (ablation on ScanNet only, no consideration on padding)
            # with self.profiler.record_function('LoFTR/coarse-matching/sim-matrix'):
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            if mask_c0 is not None:
                valid_sim_mask = mask_c0[..., None] * mask_c1[:, None]
                _inf = torch.zeros_like(sim_matrix)
                _inf[~valid_sim_mask.bool()] = -1e9
                del valid_sim_mask
                sim_matrix += _inf
            # with self.profiler.record_function('LoFTR/coarse-matching/dual-softmax'):
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
            if self.config['spg_spvs']:
                data.update({'conf_matrix_with_bin': conf_matrix})
        
        elif self.type == 'sinkhorn':
            # sinkhorn, dustbin included
            with amp.autocast(enabled=self.skh_enable_fp16):
                log_assign_matrix = self.optimal_transport(feat_c0, feat_c1, data, mask_c0, mask_c1)
                assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]
            # logger.debug(f"min(P)={conf_matrix.min()} | max(P)={conf_matrix.max()}")

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                # FIXME: Intensive mem usage caused by mask-select (https://github.com/pytorch/pytorch/issues/30246)
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0
                
            if self.config['spg_spvs']:
                data.update({'conf_matrix_with_bin': assign_matrix.clone()})
        data.update({'conf_matrix': conf_matrix})

        # predict coarse matches from conf_matrix
        with self.profiler.record_function('LoFTR/coarse-matching/get_coarse_match'):
            data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {'h0c': data['hw0_c'][0], 'w0c': data['hw0_c'][1],
                        'h1c': data['hw1_c'][0], 'w1c': data['hw1_c'][1]}
        device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)', **axes_lengths)

        # 2. mutual nearest
        mask = mask \
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        if self.fine_detector == 'OnGrid' or self.guided_matching:
            if self.config['_gt']:
                b_ids, i_ids, j_ids = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']
                mconf = torch.ones_like(b_ids, dtype=conf_matrix.dtype)
            elif self._n_rand_samples is not None:
                b_ids = conf_matrix.new_zeros(self._n_rand_samples, dtype=torch.long)
                i_ids = torch.randint(0, conf_matrix.shape[1], (self._n_rand_samples,), dtype=torch.long, device=device)
                j_ids = torch.randint(0, conf_matrix.shape[2], (self._n_rand_samples,), dtype=torch.long, device=device)
                mconf = torch.ones_like(b_ids, dtype=conf_matrix.dtype)
            else:
                mask_v, all_j_ids = mask.max(dim=2)
                with self.profiler.record_function('LoFTR/coarse-matching/get_coarse_match/argmax-conf'):
                    b_ids, i_ids = torch.where(mask_v)
                j_ids = all_j_ids[b_ids, i_ids]
                mconf = conf_matrix[b_ids, i_ids, j_ids]
        elif self.fine_detector in ['SuperPoint', 'SuperPointEC', 'SIFT']:
            # keep the coarse matches where there is a kpt in the left coarse grid.
            scale = data['hw0_i'][0] / data['hw0_c'][0]  # c=>i scale
            detector_b_ids = data['detector_b_ids']
            detector_grids_c = (data['detector_kpts0'] / scale).round().long()
            
            # clip grid kpts for the sake of no `remove_borders` used.
            detector_grids_c[:, 0].clip_(0, data['hw0_c'][1] - 1)
            detector_grids_c[:, 1].clip_(0, data['hw0_c'][0] - 1)
            
            detector_i_ids = detector_grids_c[:, 0] + detector_grids_c[:, 1] * data['hw0_c'][1]
            # option 1: use detector_kpts that appear on MNN mask
            # TODO: Relaxing the MNN condition?
            internal_ids, j_ids = torch.where(mask[detector_b_ids, detector_i_ids])
            # ↑ existence of multiple keypoints in a single cell is allowed.
            data.update({"internal_ids": internal_ids.cpu()})
            i_ids = detector_i_ids[internal_ids]
            b_ids = detector_b_ids[internal_ids]
            mconf = conf_matrix[b_ids, i_ids, j_ids]
            i_associated_kpts = data['detector_kpts0'][internal_ids]  # input resolution kpts
        # elif self.fine_detector in ["SuperPoint and grid"]:
        elif "and grid" in self.fine_detector:  # TODO: The code might be over-complex
        # means that if there are spp keypoints around grid points, use spp keypoints instead of gird points.
            # spp matches finding
            scale = data['hw0_i'][0] / data['hw0_c'][0]
            detector_b_ids = data['detector_b_ids']
            detector_grids_c = (data['detector_kpts0'] / scale).round().long()

            # clip grid kpts for the sake of no `remove_borders` used.
            detector_grids_c[:, 0].clip_(0, data['hw0_c'][1] - 1)
            detector_grids_c[:, 1].clip_(0, data['hw0_c'][0] - 1)
            
            detector_i_ids = detector_grids_c[:, 0] + detector_grids_c[:, 1] * data['hw0_c'][1]

            # find green points
            internal_ids, spp_j_ids = torch.where(mask[detector_b_ids, detector_i_ids])
            data.update({"internal_ids": internal_ids.cpu()})
            spp_i_ids = detector_i_ids[internal_ids]
            spp_b_ids = detector_b_ids[internal_ids]
            spp_mconf = conf_matrix[spp_b_ids, spp_i_ids, spp_j_ids]
            i_associated_kpts_spp = data['detector_kpts0'][internal_ids]

            # grid points finding
            mask_v, all_j_ids = mask.max(dim=2)
            spp_identifier = torch.ones_like(mask_v, device=device)
            spp_identifier[spp_b_ids, spp_i_ids] = 0  # 0 means occupied by spp points,1 means available

            grid_b_ids, grid_i_ids = torch.where(mask_v)
            grid_j_ids = all_j_ids[grid_b_ids, grid_i_ids]
            grid_mconf = conf_matrix[grid_b_ids, grid_i_ids, grid_j_ids]
            
            spp_identifier_mask = spp_identifier[grid_b_ids, grid_i_ids]

            # used to find out LoFTR grid points which conflict with detector's keypoints, used to dump only 
            grid_b_ids_overlap, grid_i_ids_overlap, grid_j_ids_overlap = map(
                lambda y: y[~spp_identifier_mask],
                [grid_b_ids,grid_i_ids,grid_j_ids])

            # filter spp occupied points
            grid_b_ids, grid_i_ids, grid_j_ids, grid_mconf = map(
                lambda y: y[spp_identifier_mask],
                [grid_b_ids, grid_i_ids, grid_j_ids, grid_mconf])
            i_associated_kpts_grid=torch.stack(
                [grid_i_ids % data['hw0_c'][1], grid_i_ids // data['hw0_c'][1]],
                dim=1) * scale

            b_ids, i_ids, j_ids, mconf, i_associated_kpts=map(
                lambda x,y:torch.cat([x,y],dim=0),
                [spp_b_ids, spp_i_ids, spp_j_ids, spp_mconf, i_associated_kpts_spp],
                [grid_b_ids, grid_i_ids, grid_j_ids, grid_mconf, i_associated_kpts_grid])

            detector_kpts_mask = torch.cat([torch.ones_like(spp_b_ids,device=device),torch.zeros_like(grid_b_ids,device=device)],dim=0) # only used for test_dump

        else:
            raise NotImplementedError

        # 5. when TRAINING
        # select only part of coarse matches for fine-level training
        # pad with gt coarses matches
        if self.training:
            # NOTE: The sampling is performed across all pairs in a batch without manually balancing
            # NOTE: #samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = calc_max_candidates(data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred, (num_matches_train - self.train_pad_num_gt_min,), device=device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data['spv_b_ids']), (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),), device=device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']], [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

            # for padded gt, use grid-points
            if self.fine_detector in ['SuperPoint', 'SuperPointEC', 'SIFT'] or 'and grid' in self.fine_detector:
                # consider size of the real input image
                patch_center = torch.stack([data['spv_i_ids'] % data['hw0_c'][1],
                                            data['spv_i_ids'] // data['hw0_c'][1]], dim=1) * scale
                i_associated_kpts = torch.cat([i_associated_kpts[pred_indices], patch_center[gt_pad_indices]], dim=0)

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids][:, [1, 0]] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids][:, [1, 0]] if 'scale1' in data else scale
        mkpts0_c = torch.stack([i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]], dim=1) * scale1

        if not self.guided_matching and (self.fine_detector in ['SuperPoint', 'SuperPointEC', 'SIFT'] or 'and grid' in self.fine_detector):
            # (coarse centers => keypoints) offsets (input resolution)
            i_associated_kpts_local = i_associated_kpts - \
                torch.stack([i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]], dim=1) * scale
            coarse_matches.update({"i_associated_kpts_local": i_associated_kpts_local})
            if 'and grid' in self.fine_detector:
                coarse_matches.update({'detector_kpts_mask': detector_kpts_mask})
                # TODO: not implemented for SuperPoint & SuperPointEC
        
        if self.save_coarse_all_matches and self.fine_detector in ['SuperPoint', 'SuperPointEC','SuperPoint and grid']:
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids_coarse_full, i_ids_coarse_full = torch.where(mask_v)
            j_ids_coarse_full = all_j_ids[b_ids_coarse_full,i_ids_coarse_full]
            scale0 = scale * data['scale0'][b_ids_coarse_full][:, [1, 0]] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][b_ids_coarse_full][:, [1, 0]] if 'scale1' in data else scale
            coarse_full_match_points0 = torch.stack([i_ids_coarse_full % data['hw0_c'][1], i_ids_coarse_full // data['hw0_c'][1]], dim=1)*scale0
            coarse_full_match_points1 = torch.stack([j_ids_coarse_full % data['hw1_c'][1], j_ids_coarse_full // data['hw1_c'][1]], dim=1)*scale1
            coarse_matches.update({"coarse_full_match_b_ids":b_ids_coarse_full,'coarse_full_match_points0':coarse_full_match_points0,'coarse_full_match_points1':coarse_full_match_points1})

            # store coarse grid matches that there are detector keypoints in their's local window
            scale0 = scale * data['scale0'][grid_b_ids_overlap][:, [1, 0]] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][grid_b_ids_overlap][:, [1, 0]] if 'scale1' in data else scale 
            coarse_overlap_match_points0 = torch.stack([grid_i_ids_overlap % data['hw0_c'][1], grid_i_ids_overlap // data['hw0_c'][1]], dim=1) * scale0
            coarse_overlap_match_points1 = torch.stack([grid_j_ids_overlap % data['hw1_c'][1], grid_j_ids_overlap // data['hw1_c'][1]], dim=1) * scale1
            coarse_matches.update({"coarse_overlap_match_b_ids":grid_b_ids_overlap,"coarse_overlap_match_points0":coarse_overlap_match_points0,"coarse_overlap_match_points1":coarse_overlap_match_points1})

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches
    
    @property
    def n_rand_samples(self):
        return self._n_rand_samples
    
    @n_rand_samples.setter
    def n_rand_samples(self, value):
        logger.warning(f'Setting {type(self).__name__}.n_rand_samples to {value}.')
        self._n_rand_samples = value