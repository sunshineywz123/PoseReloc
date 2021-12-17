import torch
import numpy as np

from .unets import Unet, thin_setup
from pathlib import Path

from .utils.structs import NpArray, Features
from .utils.detector import Detector

DEFAULT_SETUP = {**thin_setup, 'bias': True, 'padding': True}


class DISK(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super(DISK, self).__init__()

        self.desc_dim = config["descriptor_dim"]
        self.kind = config["kind"]  # choice:["nms","rng"]
        self.max_keypoints = config["max_keypoints"]
        self.cut_off = config["cut_off"]
        self.window_size_nms = config["window_size_nms"]

        self.unet = Unet(
            in_features=3, size=config["kernel_size"],
            down=[16, 32, 64, 64, 64],
            up=[64, 64, 64, self.desc_dim+1],
            setup=DEFAULT_SETUP,
        )
        self.detector = Detector(window=config["window_size_rng"])

        path = Path(__file__).parent / 'disk.pth'
        state_dict = torch.load(str(path), map_location='cpu')
        if 'extractor' in state_dict:
            weights = state_dict["extractor"]
        elif 'disk' in state_dict:
            weights = state_dict['disk']
        else:
            raise KeyError("Incompatible weight file")
        self.load_state_dict(weights)

    def _split(self, unet_output):
        '''
        Splits the raw Unet output into descriptors and detection heatmap.
        '''
        assert unet_output.shape[1] == self.desc_dim + 1

        descriptors = unet_output[:, :self.desc_dim]
        heatmap = unet_output[:, self.desc_dim:]

        return descriptors, heatmap

    def forward(
        self,
        inputs,
        mode
    ):
        ''' allowed values for `kind`:
            * rng
            * nms
        '''
        images = inputs["image"]
        B = images.shape[0]
        C = images.shape[1]

        images = images.expand(-1, 3, -1, -1) if C == 1 else images
        try:
            descriptors, heatmaps = self._split(self.unet(images))
        except RuntimeError as e:
            if 'Trying to downsample' in str(e):
                msg = ('U-Net failed because the input is of wrong shape. With '
                       'a n-step U-Net (n == 4 by default), input images have '
                       'to have height and width as multiples of 2^n (16 by '
                       'default).')
                raise RuntimeError(msg) from e
            else:
                raise

        keypoints = {
            'rng': self.detector.sample,
            'nms': self.detector.nms,
        }[self.kind](heatmaps, n=self.max_keypoints, window_size=self.window_size_nms, cutoff=self.cut_off, mode=mode)

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return {"keypoints": torch.stack([feature.kp for feature in features], dim=0).long(),
                "descriptors": torch.stack([feature.desc for feature in features], dim=0).permute(0, 2, 1),
                "scores": torch.stack([feature.kp_logp for feature in features], dim=0)}
