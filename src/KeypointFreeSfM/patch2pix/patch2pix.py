import torch.nn as nn
import torch
import numpy as np
import immatch
import yaml
import os
from immatch.utils.model_helper import init_model

def parse_model_config(config_file, benchmark_name, root_dir='.'):
    # config_file = f'{root_dir}/configs/{config}.yml'
    with open(config_file, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)[benchmark_name]
        
        # Update pretrained model path
        if 'ckpt' in model_conf and root_dir != '.':
            model_conf['ckpt'] = os.path.join(root_dir, model_conf['ckpt'])
            if 'coarse' in model_conf and 'ckpt' in model_conf['coarse']:
                model_conf['coarse']['ckpt'] = os.path.join(
                    root_dir, model_conf['coarse']['ckpt']
                )
    return model_conf

class Patch2Pix(nn.Module):
    def __init__(self, type='patch2pix'):
        super().__init__()

        if type == "patch2pix":
            config_path = 'configs/patch2pix_configs/patch2pix.yml'
            # Initialize model
            with open(config_path, 'r') as f:
                args = yaml.load(f, Loader=yaml.FullLoader)['example']
            model = immatch.__dict__[args['class']](args)
        elif type == "patch2pix_superglue":
            config_path = 'configs/patch2pix_configs/patch2pix_superglue.yml'
            args = parse_model_config(config_path, 'aachen')
            model = immatch.__dict__[args['class']](args)
        else:
            raise NotImplementedError

        self.matcher = lambda im1, im2: model.match_pairs(im1, im2)
    def forward(self, batch):
        image_name0, image_name1 = batch['pair_key']
        matches, _, _, scores = self.matcher(image_name0, image_name1)
        matches, scores = map(lambda x: torch.from_numpy(x), [matches, scores])

        mkpts0_f = matches[:, :2]
        mkpts1_f = matches[:, 2:]
        m_bids = torch.zeros_like(scores)
        batch.update({'m_bids': m_bids, 'mkpts0_f': mkpts0_f, 'mkpts1_f': mkpts1_f, 'mconf': scores})