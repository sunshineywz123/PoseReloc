from sys import version
import torch
import numpy as np
import torch.nn.functional as F

from torch.distributions import Categorical, Bernoulli

from .structs import Features, NpArray
from .nms import nms


def select_on_last(values, indices):
    '''
    WARNING: this may be reinventing the wheel, but I don't know how to do
    it otherwise with PyTorch.

    This function uses an array of linear indices `indices` between [0, T] to
    index into `values` which has equal shape as `indices` and then one extra
    dimension of size T.
    '''
    return torch.gather(
        values,
        -1,
        indices[..., None]
    ).squeeze(-1)


def point_distribution(logits):
    '''
    Implements the categorical proposal -> Bernoulli acceptance sampling
    scheme. Given a tensor of logits, performs samples on the last dimension,
    returning
        a) the proposals
        b) a binary mask indicating which ones were accepted
        c) the logp-probability of (proposal and acceptance decision)
    '''

    proposal_dist = Categorical(logits=logits)
    proposals = proposal_dist.sample()
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = select_on_last(logits, proposals).squeeze(-1)

    accept_dist = Bernoulli(logits=accept_logits)
    accept_samples = accept_dist.sample()
    accept_logp = accept_dist.log_prob(accept_samples)
    accept_mask = accept_samples == 1.

    logp = proposal_logp + accept_logp

    return proposals, accept_mask, logp


class Keypoints:
    '''
    A simple, temporary struct used to store keypoint detections and their
    log-probabilities. After construction, merge_with_descriptors is used to
    select corresponding descriptors from unet output.
    '''

    def __init__(self, xys, logp):
        self.xys = xys
        self.logp = logp

    def merge_with_descriptors(self, descriptors):
        '''
        Select descriptors from a dense `descriptors` tensor, at locations
        given by `self.xys`
        '''
        x, y = self.xys.T

        desc = descriptors[:, y, x].T
        desc = F.normalize(desc, dim=-1)

        return Features(self.xys.to(torch.float32), desc, self.logp)


class Detector:
    def __init__(self, window=8):
        self.window = window

    def _tile(self, heatmap):
        '''
        Divides the heatmap `heatmap` into tiles of size (v, v) where
        v==self.window. The tiles are flattened, resulting in the last
        dimension of the output T == v * v.
        '''
        v = self.window
        b, c, h, w = heatmap.shape

        assert heatmap.shape[2] % v == 0
        assert heatmap.shape[3] % v == 0

        return heatmap.unfold(2, v, v) \
                      .unfold(3, v, v) \
                      .reshape(b, c, h // v, w // v, v*v)

    def sample(self, heatmap, **kwargs):
        '''
            Implements the training-time grid-based sampling protocol
        '''
        v = self.window
        dev = heatmap.device
        B, _, H, W = heatmap.shape

        assert H % v == 0
        assert W % v == 0

        # tile the heatmap into [window x window] tiles and pass it to
        # the categorical distribution.
        heatmap_tiled = self._tile(heatmap).squeeze(1)
        proposals, accept_mask, logp = point_distribution(heatmap_tiled)

        # create a grid of xy coordinates and tile it as well
        cgrid = torch.stack(torch.meshgrid(
            torch.arange(H, device=dev),
            torch.arange(W, device=dev),
        )[::-1], dim=0).unsqueeze(0)
        cgrid_tiled = self._tile(cgrid)

        # extract xy coordinates from cgrid according to indices sampled
        # before
        xys = select_on_last(
            self._tile(cgrid).repeat(B, 1, 1, 1, 1),
            # unsqueeze and repeat on the (xy) dimension to grab
            # both components from the grid
            proposals.unsqueeze(1).repeat(1, 2, 1, 1)
        ).permute(0, 2, 3, 1)  # -> bhw2

        keypoints = []
        for i in range(B):
            mask = accept_mask[i]
            keypoints.append(Keypoints(
                xys[i][mask],
                logp[i][mask],
            ))

        return np.array(keypoints, dtype=object)

    def nms(
        self,
        heatmap,
        n=None,
        **kwargs
    ):
        '''
            Inference-time nms-based detection protocol
        '''
        heatmap = heatmap.squeeze(1)
        nmsed = nms(heatmap, kwargs["window_size"], kwargs["cutoff"])

        keypoints = []
        for b in range(heatmap.shape[0]):
            yx = nmsed[b].nonzero(as_tuple=False)
            logp = heatmap[b][nmsed[b]]
            xy = torch.flip(yx, (1, ))

            if n is not None:
                xy, logp = top_k_keypoints(
                    xy, logp, n, img_h=heatmap.shape[-2], img_w=heatmap.shape[-1], mode=kwargs["mode"])
                '''
                n_ = min(n+1, logp.numel())

                #superglue constrains
                assert n+1<logp.numel(),f"not enough points,want:{n} have:{logp.numel()}"

                # torch.kthvalue picks in ascending order and we want to pick in
                # descending order, so we pick n-th smallest among -logp to get
                # -threshold
                minus_threshold, _indices = torch.kthvalue(-logp, n_)
                mask = logp > -minus_threshold    #may be problem: same score!
                
                if mask.float().sum() != n_-1:
                    mask_equal = logp == -minus_threshold
                    assert mask_equal.float().sum()!=1,"num of threshold is 1"
                    diff=n_-1-mask.float().sum()
                    assert mask_equal.float().sum()>=diff,"num of threhold smaller than diff"
                    for i in range(mask_equal.numel()):
                        if mask_equal[i]==True:
                            if diff!=0:
                                diff-=1
                            else:
                                mask_equal[i]=False
                    mask=mask | mask_equal
                    assert mask.float().sum() == n,"still not equal"


                xy   = xy[mask]
                logp = logp[mask]
                '''

            keypoints.append(Keypoints(xy, logp))

        return np.array(keypoints, dtype=object)


def pad_keypoints_random_v2(keypoints, scores, img_h: int, img_w: int, n_target_kpts: int):
    """ Pad the given keypoints to the target #kpts. The padded kpts shouldn't overlap with
    existing kpts.
    Args:
        keypoints (torch.Tensor): sorted keypoints with shape (n_kpts, 2). 
            (sorted is not required)
    Returns:
        padded_kpts (torch.Tensor): (n_target_kpts, 2).
        padded_scores (torch.Tensor): (n_target_kpts,)
    """
    device = keypoints.device
    dtype = keypoints.dtype
    n_pad = n_target_kpts - keypoints.shape[0]
    # TODO: Optimization
    while n_pad > 0:
        # TODO: add torch.Generator
        rand_kpts_x = torch.randint(0, img_w, (n_pad, ), dtype=dtype, device=device)
        rand_kpts_y = torch.randint(0, img_h, (n_pad, ), dtype=dtype, device=device)
        rand_kpts = torch.stack([rand_kpts_y, rand_kpts_x], 1)

        exist = (rand_kpts[:, None, :] == keypoints[None, :, :]).all(-1).any(1)  # (n_pad, )
        kept_kpts = rand_kpts[~exist]  # (n_kept, 2)
        n_pad -= len(kept_kpts)

        if len(kept_kpts) > 0:
            keypoints = torch.cat([keypoints, kept_kpts], 0)
            scores = torch.cat([scores, torch.zeros(len(kept_kpts), dtype=scores.dtype, device=device)], 0)
    return keypoints, scores


def top_k_keypoints(keypoints, scores, k: int, img_h: int, img_w: int, mode: str):
    """
    Args:
        keypoints (torch.Tensor): (n_kpts, 2)
        scores (torch.Tensor): (n_kpts, )
    """
    if k >= len(keypoints):
        print(f"warning!keypoints are not enough,want {k}, only have {len(keypoints)}, will padding!")
        # Randomly pad keypoints to k with score = 0
        if mode == 'train':
            padded_kpts, padded_scores = pad_keypoints_random_v2(keypoints, scores, img_h, img_w, k)
            return padded_kpts, padded_scores
        else:
            return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores
