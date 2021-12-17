"""
Guided matching based on fine-level LoFTR matches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .guided_matching import GuidedMatching as GuidedMatchingCoarse


def build_guided_matching(config):
    if config['enable']:
        return GuidedMatching(config)
    else:
        return PlaceHolder()


class PlaceHolder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, data):
        return data


def match_mkpts_and_kpts(mkpts, kpts, mutual_check=True):
    dist_mat = torch.norm(mkpts[:, None] - kpts[None], p=2, dim=-1)
    v_m2k, i_m2k = torch.min(dist_mat, -1)
    valid = torch.ones_like(v_m2k, dtype=torch.bool)
    if not mutual_check:
        return v_m2k, i_m2k, valid
    
    v_k2m, i_k2m = torch.min(dist_mat, 0)
    m_inds = torch.arange(len(mkpts), device=mkpts.device)
    loop = torch.gather(i_k2m, -1, i_m2k)
    mnn = valid & (m_inds == loop)
    return v_m2k, i_m2k, mnn
    

def build_mkpts_to_kpts_mapper(bids, kpts, mbids, mkpts, scale_i2o, win_size, B, device):
    assert B == 1 and (bids == 0).all(), "batch-size != 1 not implemented yet."
    o_kpts = kpts * scale_i2o
    
    mkpts_to_kpts_mapper = [[] for _ in range(len(mkpts))]
    # dist_mat = torch.norm(mkpts[:, None] - o_kpts[None], p=2, dim=-1)
    # take mutual nearest neighbor
    # min_dist, arg_min_kpts = torch.min(dist_mat, -1)  # TODO: use top-k instead of min
    min_dist, arg_min_idx, valid = match_mkpts_and_kpts(mkpts, o_kpts, mutual_check=True)
    valid_kpts = valid & (min_dist <= win_size // 2)
    for m_id in range(len(mkpts)):
        if valid_kpts[m_id]:
            mkpts_to_kpts_mapper[m_id].append(arg_min_idx[m_id])
    return mkpts_to_kpts_mapper


class GuidedMatching(GuidedMatchingCoarse):
    def __init__(self, config):
        super().__init__(config)
        self.keep_refined_pts = config['keep_refined_pts']
    
    def forward(self, data):
        B, _device, _dtype = data['image0'].shape[0], data['image0'].device, data['image0'].dtype
        
        # 1. build mkpts1_f => kpts1 mapper
        scale_i2o_1 = 1 if 'scale1' not in data else data['scale1'][:, [1, 0]]
        f_to_kpts_mapper = build_mkpts_to_kpts_mapper(
            data['detector_b_ids1'], data['detector_kpts1'],
            data['m_bids'], data['mkpts1_f'], scale_i2o_1,
            self.win_size, B, _device
        )
        
        # 2. handle multi-kpts-in-one-win (not needed yet)
        
        # 3. handle no-kpts-in-one-win (1.discard 2. keep the fine-level refinement results)
        f_to_kpts_mapper = self.handle_no_kpt_in_one(f_to_kpts_mapper)
        
        # 4. update data
        m_bids, m_kpts0_f, m_kpts1_f, m_conf = [], [], [], []
        for m_id, kpt_ids1 in enumerate(f_to_kpts_mapper):
            # if len(kpt_ids1) != 0:
            assert len(kpt_ids1) <= 1
            for id1 in kpt_ids1:
                b_id1 = data['detector_b_ids1'][id1]
                m_bids.append(b_id1)
                m_conf.append(data['mconf'][m_id])
                m_kpts0_f.append(data['mkpts0_f'][m_id])
                m_kpts1_f.append(data['detector_kpts1'][id1] * scale_i2o_1[b_id1])
            if len(kpt_ids1) == 0 and self.keep_refined_pts:
                m_bids.append(data['m_bids'][m_id])
                m_conf.append(data['mconf'][m_id])
                m_kpts0_f.append(data['mkpts0_f'][m_id])
                m_kpts1_f.append(data['mkpts1_f'][m_id])
        data.update({
            "m_bids": data['m_bids'].new_tensor(m_bids),
            "mconf": data['mconf'].new_tensor(m_conf),
            "mkpts0_f": torch.stack(m_kpts0_f, 0).to(data['mkpts0_c'])
                        if len(m_kpts0_f) > 0 else data['mkpts0_c'].new_empty((0, 2)),
            "mkpts1_f": torch.stack(m_kpts1_f, 0).to(data['mkpts1_c'])
                        if len(m_kpts1_f) > 0 else data['mkpts1_c'].new_empty((0, 2))
        })

    def handle_no_kpt_in_one(self,
                             f_to_kpts_mapper):
        if self.no_kpt_in_one_win == 'discard':
            for i in range(len(f_to_kpts_mapper)):
                if len(f_to_kpts_mapper[i]) == 0:
                    f_to_kpts_mapper[i] = []
            return f_to_kpts_mapper
        elif self.no_kpt_in_one_win == 's2d':
            raise NotImplementedError()
        else:
            raise ValueError()
