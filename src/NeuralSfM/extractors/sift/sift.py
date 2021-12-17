"""
A wrapper upon the opencv sift keypoint.
"""

import cv2
from joblib import Parallel, delayed

import torch
import numpy as np
from torch import nn

from ..base import BaseModel


def _sift_detect(sift, img: np.ndarray):
    assert img.ndim == 2
    gray = np.uint8(img)
    keypoints = sift.detect(gray, None)
    
    response = np.array([k.response for k in keypoints])
    sort_idx = np.argsort(response)[::-1]  # descent
    keypoint = np.array([k.pt for k in keypoints])[sort_idx]
    size = np.array([k.size for k in keypoints])[sort_idx]
    angle = np.array([k.angle for k in keypoints])[sort_idx]
    response = np.array([k.response for k in keypoints])[sort_idx]
    return keypoint, size, angle, response


lower_threshold_conf = {
    "contrastThreshold": -10000,
    "edgeThreshold": -10000,
}

class SIFT(BaseModel):
    default_conf = {  # default opencv sift config
        "nfeatures": 0,
        "nOctaveLayers": 3,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6,
        # "descriptorType": cv2.CV_32F
    }
    required_data_keys = ['image']
    
    def _init(self, config):
        self.sift = cv2.SIFT_create(**config)
        
    def _forward(self, data, mode='eval'):
        assert mode == 'eval', 'training mode not supported!'
        images = data['image']  # (B, C, H, W)
        assert 0 <= images.min() <= 1, 'The input images are assumed normlized to range (0, 1)'
        assert images.shape[1] == 1, 'Only grayscale images supported!'
        _bs, _device, _dtype = data['image'].shape[0], data['image'].device, data['image'].dtype
        _new_tensor = lambda x: torch.tensor(x, device=_device, dtype=_dtype)

        images_np = (images * 255).permute(0, 2, 3, 1).cpu().numpy()
        
        if _bs > 1:
            detections = Parallel()(
                delayed(lambda x: _sift_detect(self.sift, x[..., 0]))(img)
                for img in images_np)
        else:
            detections = [_sift_detect(self.sift, images_np[0, ..., 0])]
        
        keypoints = [_new_tensor(d[0]) for d in detections]  # [B, (N, 2)]
        scores = [_new_tensor(d[3]) for d in detections]  # [B, (N, )]
        
        return {
            'keypoints': keypoints,
            'scores': scores
        }
