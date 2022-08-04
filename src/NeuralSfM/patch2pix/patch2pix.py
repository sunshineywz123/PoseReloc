from turtle import forward
import torch.nn as nn
import torch
import numpy as np
import immatch
import yaml

class Patch2Pix(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize model
        with open('configs/patch2pix_configs/patch2pix.yml', 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['example']
        model = immatch.__dict__[args['class']](args)
        self.matcher = lambda im1, im2: model.match_pairs(im1, im2)
    def forward(self, batch):
        image_name0, image_name1 = batch['pair_key']
        matches, _, _, scores = self.matcher(image_name0, image_name1)
        matches, scores = map(lambda x: torch.from_numpy(x), [matches, scores])

        mkpts0_f = matches[:, :2]
        mkpts1_f = matches[:, 2:]
        m_bids = torch.zeros_like(scores)
        batch.update({'m_bids': m_bids, 'mkpts0_f': mkpts0_f, 'mkpts1_f': mkpts1_f, 'mconf': scores})