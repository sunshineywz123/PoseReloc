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
# agreements explicitly covering such access.
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

import torch
from pathlib import Path
from torch import nn

from .base_model import BaseModel
from kornia.utils import create_meshgrid
from .superpoint import simple_nms, remove_borders, top_k_keypoints


class SuperPointEC(BaseModel):
    """ SuperPoint on Every Cell, a special version for loftr """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    required_data_keys = ['image']

    def _init(self, config):
        self.config = {**self.default_config, **config}

        self.align_center_with_resnet = self.config['align_center_with_resnet']
        self.ec_version = self.config['version']
        """
        version definition:
            v1: first remove dustbin and then pick the highest position.  
            v2: first remove dustbin and then use simple-nms, do not perform thresholding.
        """

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
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPointEC model')

    def _forward(self, data, mode='eval'):
        """ Compute keypoints, scores, descriptors for image """
        # assert mode in ['train', 'eval']
        assert mode == 'eval', "superpoint-ec can only be run in evaluation mode"

        image = data['image']
        H, W = image.shape[-2:]

        # borders of ResNet 1/8 feature map won't correspond to a superpoint response map.
        # This is not perfect, better using a resnet-spp. (cannot exactly align VGG & ResNet)
        if self.align_center_with_resnet:
            b, _, h, w = image.shape
            assert (h % 8 == 0) and (w % 8 == 0)
            image = image[:, :, 4:-4, 4:-4]

        # Shared Encoder (for interest point decoder & descriptor decoder)
        # s0
        x = self.relu(self.conv1a(image))
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
        scores = torch.nn.functional.softmax(scores, 1)
        b, _, h, w = scores.shape

        if self.ec_version == 'v1':  # first remove dustbin and then pick the highest position.
            # remove dustbin
            scores = scores[:, :-1]  # (N, 65->64, H/8, W/8)
            grids = create_meshgrid(h, w, False, device=scores.device) * 8  # [1, h, w, 2]
            # pick the highest position
            scores, indices = scores.max(1)
            keypoints = torch.stack([grids[..., 0] + indices % 8,
                                     grids[..., 1] + indices // 8], dim=-1)
            keypoints = keypoints[..., [1, 0]]  # <x, y> => <y, x>
            if self.align_center_with_resnet:
                keypoints += 4
            keypoints = [k_.reshape(-1, 2) for k_ in keypoints]
            scores = [s_.reshape(-1) for s_ in scores]
        elif self.ec_version == 'v2':
            # cannot guarantee there is only 1 point in each cell!
            scores = scores[:, :-1]  # (N, 65->64, H/8, W/8)
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
            scores = simple_nms(scores, self.config['nms_radius'])  # (N, H, W)
            # keypoints = [torch.nonzero(s, as_tuple=False).flip(1) for s in scores]  # [N, (n_kpts, 2)] - <x, y>
            keypoints = [torch.nonzero(s, as_tuple=False) for s in scores]  # [N, (n_kpts, 2)] - <y, x>
            scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]  # [N, (n_kpts,)] - traverse along batch dim
            if self.align_center_with_resnet:
                for kps_ in keypoints:
                    kps_ += 4
        else:
            raise NotImplementedError

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], H, W)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'], H, W, mode)
                for k, s in zip(keypoints, scores)]))  # [N, (max_kpts, 2)], [N, (max_kpts, )] if train
        
        keypoints = [torch.flip(k, [1]) for k in keypoints]  # <y, x> => <x, y>
        
        return {
            'keypoints': keypoints,  # [N, (n_kpts, 2)] - (x, y) / (N, max_kpts, 2)
            'scores': scores,  # [N, (n_kpts,)] - scores don't sum up to 1.0 / (N, max_kpts, )
        }
