import torch.nn as nn
import torch
import numpy as np
from .tools import ImgMatcher
from .normalization import normalize_float

class DRCNet(nn.Module):
    def __init__(self, checkpoint, use_cuda, half_precision):
        super().__init__()
        self.matcher = ImgMatcher(checkpoint, use_cuda, half_precision)
        self.num_pts = 2000
    
    def forward(self, batch):
        hA, wA = batch['image0'].shape[-2:]
        hB, wB = batch['image1'].shape[-2:]

        src = batch['image0'].expand(-1, 3, -1, -1)
        src = normalize_float(src)
        tgt = batch['image1'].expand(-1, 3, -1, -1)
        tgt = normalize_float(tgt)
        result, scores, features = self.matcher({'source_image':src, 'target_image':tgt}, num_pts=self.num_pts, central_align=False)

        corr4d, featureA_0, featureB_0 = features
        fs1, fs2 = featureA_0.shape[2:]
        fs3, fs4 = featureB_0.shape[2:]
        ratio = int(np.round(hA/fs1))
        xA_ = result[:, 0] / ratio
        yA_ = result[:, 1] / ratio
        xB_ = result[:, 2] / ratio
        yB_ = result[:, 3] / ratio
        score_ = scores

        YA,XA=torch.meshgrid(torch.arange(fs1),torch.arange(fs2))
        YB,XB=torch.meshgrid(torch.arange(fs3),torch.arange(fs4))

        YA = YA.contiguous()
        XA = XA.contiguous()
        YB = YB.contiguous()
        XB = XB.contiguous()

        YA=(YA+0.5)/(fs1)*hA
        XA=(XA+0.5)/(fs2)*wA
        YB=(YB+0.5)/(fs3)*hB
        XB=(XB+0.5)/(fs4)*wB

        XA = XA.view(-1).data
        YA = YA.view(-1).data
        XB = XB.view(-1).data
        YB = YB.view(-1).data

        keypoints_A=torch.stack((XA,YA),dim=1).to(yB_.device)
        keypoints_B=torch.stack((XB,YB),dim=1).to(yB_.device)

        idx_A = (yA_*fs2+xA_).long().view(-1,1)
        idx_B = (yB_*fs4+xB_).long().view(-1,1)
        score = score_.view(-1,1)

        # change to loftr format:
        mkpts0_f = keypoints_A[idx_A[:, 0]]
        mkpts1_f = keypoints_B[idx_B[:, 0]]
        m_bids = torch.zeros_like(score)

        # if idx_A.shape[0] == 0:
        #     mkpts0_f = torch.empty((0,2))
        #     mkpts1_f = torch.empty((0,2))
        #     m_bids = torch.empty((0,1))
        #     score = torch.emtpy((0,1))
        # else:

        batch.update({'m_bids': m_bids, 'mkpts0_f': mkpts0_f, 'mkpts1_f': mkpts1_f, 'mconf': score[:, 0]})
