import cv2
import numpy as np
import torch.nn as nn

class SIFT(nn.Module):
    default_conf = {
        'keypoints_threshold': 0.02
    }
    
    def __init__(self, conf=None):
        super().__init__()
        if conf:
            self.conf = conf = {**self.default_conf, **conf}
        else:
            self.conf = self.default_conf
        
        self.sift = cv2.xfeatures2d.SIFT_create()
        
    def forward(self, image):
        image_size = image.shape
        kpts_, descriptors = self.sift.detectAndCompute(image, None)

        kpts = np.array([(kpt_.pt[0], kpt_.pt[1]) for kpt_ in kpts_])
        scores = np.array([kpt_.response for kpt_ in kpts_])
        
        valid = scores > self.conf['keypoints_threshold']
        kpts, scores ,descriptors = kpts[valid], scores[valid], descriptors[valid]
        descriptors = descriptors.transpose(1, 0) / 256.
        
        return {
            'keypoints': kpts,
            'scores': scores,
            'descriptors': descriptors,
            'image_size': image_size
        }