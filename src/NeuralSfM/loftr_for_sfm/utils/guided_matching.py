"""
Guided matching based on coarse-level LoFTR matches.
"""
import torch
import numpy as np
from torch import nn


def build_coarse_to_kpts_mapper(bids, kpts, mbids, mkpts, scale_i2c, scale_o2c, b, h_c, w_c, device):
    c_mkpts = (mkpts * scale_o2c).round().long()  # no need to round
    c_kpts = (kpts * scale_i2c).round().long()
    # clip kpts for the sake of no remove_border() used
    c_kpts[:, 0].clip_(min=0, max=w_c-1)
    c_kpts[:, 1].clip_(min=0, max=h_c-1)
        
    c_to_kpts_mapper = [[[[] for _w in range(w_c)] for _h in range(h_c)] for _b in range(b)]
    matchable = torch.zeros((b, h_c, w_c), device=device, dtype=torch.bool)
    matchable[mbids, c_mkpts[:, 1], c_mkpts[:, 0]] = 1
    kpts_matchable = matchable[bids, c_kpts[:, 1], c_kpts[:, 0]]
    for b_id, kpt_id, c_kpt, m in zip(bids, range(len(kpts)), c_kpts, kpts_matchable):
        if m:
            c_to_kpts_mapper[b_id][c_kpt[1]][c_kpt[0]].append(kpt_id)
    c_to_kpts_mapper = [c_to_kpts_mapper[b_id][c_mkpt[1]][c_mkpt[0]]
                            for b_id, c_mkpt in zip(mbids, c_mkpts)]
    return c_to_kpts_mapper


class GuidedMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.win_size = config['window_size']  # not used currently
        self.multi_kpts_in_one_win = config['multi_kpts_in_one']  # ('top-1', 'mnn')
        self.no_kpt_in_one_win = config['no_kpt_in_one_win']  # ('discard', 's2d')
        
    def forward(self, data, feat_f0=None, feat_f1=None):
        """
        Args:
            data (dict): {
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M],
                'detector_kpts0': [M0, 2],
                'detector_scores0': [M0],
                'detector_descs0': [M0, C],
                'detector_b_ids0': [M0],
                'detector_kpts1': [M1, 2],
                'detector_scores1': [M1],
                'detector_descs1': [M1, C],
                'detector_b_ids1': [M1]
            }
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
        Update:
            data (dict): {
                'mkpts0_f' (torch.Tensor): [M, 2] - original image scale
                'mkpts1_f' (torch.Tensor): [M, 2]
            }
        """
        B, _device, _dtype = data['image0'].shape[0], data['image0'].device, data['image0'].dtype
        scale_i2c = data['hw0_c'][0] / data['hw0_i'][0]
        scale_i2o_0 = 1 if 'scale0' not in data else data['scale0'][:, [1, 0]]
        scale_i2o_1 = 1 if 'scale1' not in data else data['scale1'][:, [1, 0]]
        scale_o2c_0 = scale_i2c / scale_i2o_0
        scale_o2c_1 = scale_i2c / scale_i2o_1
        # 1. build coarse grid to keypoints (id) mapper
        c_to_kpts_mapper0 = build_coarse_to_kpts_mapper(
            data['detector_b_ids0'], data['detector_kpts0'],
            data['m_bids'], data['mkpts0_c'], scale_i2c,
            scale_o2c_0, B, data['hw0_c'][0], data['hw0_c'][1], _device)
        c_to_kpts_mapper1 = build_coarse_to_kpts_mapper(
            data['detector_b_ids1'], data['detector_kpts1'],
            data['m_bids'], data['mkpts1_c'], scale_i2c,
            scale_o2c_1, B, data['hw1_c'][0], data['hw1_c'][1], _device)
        
        # 2. handle multi-kpts-in-one-coarse-grid
        c_to_kpts_mapper0, c_to_kpts_mapper1 = self.handle_multi_in_one(
            data, c_to_kpts_mapper0, c_to_kpts_mapper1)
        
        # 3. handle no-kpts-in-at-least-one-coarse-grid
        c_to_kpts_mapper0, c_to_kpts_mapper1 = self.handle_no_kpt_in_one(
            data, c_to_kpts_mapper0, c_to_kpts_mapper1)
        
        # 4. update data
        m_bids, m_kpts0_f, m_kpts1_f, m_conf = [], [], [], []
        for m_id, (kpt_ids0, kpt_ids1) in enumerate(zip(c_to_kpts_mapper0, c_to_kpts_mapper1)):
            assert len(kpt_ids0) == len(kpt_ids1)
            # if len(kpt_ids0) != 0:
            for id0, id1 in zip(kpt_ids0, kpt_ids1):
                b_id0, b_id1 = data['detector_b_ids0'][id0], data['detector_b_ids1'][id1]
                assert b_id0 == b_id1
                m_bids.append(b_id0)
                m_conf.append(data['mconf'][m_id])
                m_kpts0_f.append(data['detector_kpts0'][id0] * scale_i2o_0[b_id0])
                m_kpts1_f.append(data['detector_kpts1'][id1] * scale_i2o_1[b_id1])
        data.update({
            "m_bids": data['m_bids'].new_tensor(m_bids),
            "mconf": data['mconf'].new_tensor(m_conf),
            "mkpts0_f": torch.stack(m_kpts0_f, 0).to(data['mkpts0_c'])
                        if len(m_kpts0_f) > 0 else data['mkpts0_c'].new_empty((0, 2)),
            "mkpts1_f": torch.stack(m_kpts1_f, 0).to(data['mkpts1_c'])
                        if len(m_kpts1_f) > 0 else data['mkpts1_c'].new_empty((0, 2))
        })
    
    def handle_multi_in_one(self,
                            data,
                            c_to_kpts_mapper0,
                            c_to_kpts_mapper1):
        if self.multi_kpts_in_one_win == 'top-1':
            def _filter_kpts(kpts_mapper, kpts_scores):
                def _select_top1(kpt_ids):
                    if len(kpt_ids) <= 1:
                        return kpt_ids
                    else:
                        scores = [kpts_scores[i] for i in kpt_ids]
                        return [kpt_ids[np.argmax(scores)]]
                kpts_mapper = list(map(_select_top1, kpts_mapper))
                return kpts_mapper
                            
            c_to_kpts_mapper0 = _filter_kpts(c_to_kpts_mapper0, data['detector_scores0'])
            c_to_kpts_mapper1 = _filter_kpts(c_to_kpts_mapper1, data['detector_scores1'])
            return c_to_kpts_mapper0, c_to_kpts_mapper1
        elif self.multi_kpts_in_one_win == 'mnn':
            raise NotImplementedError()
        else:
            raise ValueError()
        
    def handle_no_kpt_in_one(self,
                             data,
                             c_to_kpts_mapper0,
                             c_to_kpts_mapper1):
        assert len(c_to_kpts_mapper0) == len(c_to_kpts_mapper1)
        # mconf = data['mconf']
        
        if self.no_kpt_in_one_win == 'discard':
            for i in range(len(c_to_kpts_mapper0)):
                if len(c_to_kpts_mapper0[i]) == 0:
                    c_to_kpts_mapper1[i] = []
                if len(c_to_kpts_mapper1[i]) == 0:
                    c_to_kpts_mapper0[i] = []
            return c_to_kpts_mapper0, c_to_kpts_mapper1
        elif self.no_kpt_in_one_win == 's2d':
            # s2d with coarse grid points / kpts in one window (left=>right s2d / right=>left s2d)
            raise NotImplementedError()
        else:
            raise ValueError()
