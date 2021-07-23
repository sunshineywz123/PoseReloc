""" SuperPoint extractor for both training and inference.
Adapted from the original implementation of SuperPoint.
Training mode and evaluation mode have different output formats.
"""
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicit ly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
from time import perf_counter

import torch
from pathlib import Path
from torch import nn

from .base_model import BaseModel


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):  # keep the `scores` shape unchanged
        """ Suppress points whose score isn't the maximum within the local patch.
        """
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)  # max: 1, non-max: 0
    for _ in range(2):  # ???
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]

def pad_keypoints_random(keypoints, scores, img_h: int, img_w: int, n_target_kpts: int):
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
        kept_kpts = []
        for rand_kpt in rand_kpts:  # This loop should be vectorized
            exist = (rand_kpt == keypoints).all(1).any()
            if not exist:
                n_pad -= 1
                kept_kpts.append(rand_kpt)
        if len(kept_kpts) > 0:
            keypoints = torch.cat([keypoints, torch.stack(kept_kpts, 0)], 0)
            scores = torch.cat([scores, torch.zeros(len(kept_kpts), dtype=scores.dtype, device=device)], 0)
    return keypoints, scores

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
        n_pad -=len(kept_kpts)
        
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
        # Randomly pad keypoints to k with score = 0
        if mode == 'train':
            padded_kpts, padded_scores = pad_keypoints_random_v2(keypoints, scores, img_h, img_w, k)
            return padded_kpts, padded_scores
        else:
            return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5  # calc down-sampled keypoints positions
    keypoints /= torch.tensor([(w-1)*s, (h-1)*s]).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(BaseModel):
    """ SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    required_data_keys = ['image']

    def _init(self, config):
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # interest point decoder - all on s3 (no upsampling)
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # descriptor decoder - all on s3 (no upsampling)
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path),map_location='cpu'))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def _forward(self, data, mode='train', normalize_feats=True, ret_featmaps=False, real_hw=None):
        """ Compute keypoints, scores, descriptors for image """
        assert mode in ['train', 'eval']
        ret_dict = {}
        # Shared Encoder (for interest point decoder & descriptor decoder)
        # s0
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        # s1
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        # s2
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        # s3
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Interest Point Decoder - Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        # - softmax and remove dustbin
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]  # (N, 64, H/8, W/8)
        b, _, h, w = scores.shape
        # - pixel shuffle
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)

        scores = simple_nms(scores, self.config['nms_radius'])  # (N, H, W)

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'], as_tuple=False)
            for s in scores]  # [N, (n_kpts, 2)]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]  # [N, (n_kpts,)] - traverse along batch dim
        # scores2 = [s[k[:, 0], k[:, 1]] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        image_h = [h * 8] * len(keypoints) if real_hw is None else real_hw[0]
        image_w = [w * 8] * len(keypoints) if real_hw is None else real_hw[1]
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], int(image_h[i]), int(image_w[i]))
            for i, (k, s) in enumerate(zip(keypoints, scores))]))
        
        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'], int(image_h[i]), int(image_w[i]), mode)
                for i, (k, s) in enumerate(zip(keypoints, scores))]))  # [N, (max_kpts, 2)], [N, (max_kpts, )] if train
        
        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        if normalize_feats:
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)  # (N, D, H/8, W/8)
        if ret_featmaps:
            ret_dict.update({"featmaps": descriptors})
            
        if mode == 'eval':
            # Convert (h, w) to (x, y)
            keypoints = [torch.flip(k, [1]) for k in keypoints]  # don't cast to float here

            # Extract descriptors
            descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                                for k, d in zip(keypoints, descriptors)]  # [N, (D, n_kpts)]
        
        if mode == 'train':
            keypoints = torch.stack(keypoints, 0)  # (N, max_kpts, 2) - don't cast to float here
            scores = torch.stack(scores, 0)  # (N, max_kpts, )
            # Convert (h, w) to (x, y)
            keypoints = torch.flip(keypoints, [2])
            # Extract descriptors
            descriptors = sample_descriptors(keypoints, descriptors, 8)  # (N, D, max_kpts)

        ret_dict.update({
            'keypoints': keypoints,  # [N, (n_kpts, 2)] - (x, y) / (N, max_kpts, 2)
            'scores': scores,  # [N, (n_kpts,)] - scores don't sum up to 1.0 / (N, max_kpts, )
            'descriptors': descriptors,  # [N, (D, n_kpts)] / (N, D, max_kpts)
        })
        return ret_dict