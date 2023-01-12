import hydra
import os
import glob

import os.path as osp
import numpy as np
import natsort

from loguru import logger
from pathlib import Path
from omegaconf import DictConfig

from src.sfm_utils import pairs_from_poses_hloc, pairs_exhaustive_all

def align(cfg):
    data_dirs = cfg.dataset.data_dir
    if cfg.full_img:
        img_type = "color_full"
    else:
        img_type = "color"

    for data_dir in data_dirs:
        logger.info(f'Processing {data_dir}.')
        root_dir, sub_dirs = data_dir.split(' ')[0], data_dir.split(' ')[1:]

        img_lists = []
        for sub_dir in sub_dirs:
            seq_dir = osp.join(root_dir, sub_dir)
            img_lists += glob.glob(str(Path(seq_dir)) + '/{}/*.png'.format(img_type), recursive=True)
        img_lists = natsort.natsorted(img_lists)[::cfg.downsample_ratio]

        if len(img_lists) == 0:
            logger.info(f'No PNG image in {root_dir}')
            continue
        
        suffix = cfg.suffix
        obj_name = root_dir.split('/')[-1]
        outputs_dir_root = cfg.dataset.outputs_dir.format(obj_name) + suffix

        align_core(cfg, img_lists, outputs_dir_root)


def align_core(cfg, img_lists, outputs_dir_root):
    from src.sfm_utils import extract_features, generate_empty, \
                         match_features, triangulation, global_ba
    from src.sfm_utils.postprocess.ba_postprocess import parse_align_pose
    
    outputs_dir = osp.join(outputs_dir_root, 'outputs_' + cfg.network.detection + '_' + cfg.network.matching)
    
    feature_out = osp.join(outputs_dir, f'feats-{cfg.network.detection}.h5')
    covis_num = cfg.sfm.covis_num
    covis_pairs_out = osp.join(outputs_dir, f'pairs-covis{covis_num}.txt')
    matches_out = osp.join(outputs_dir, f'matches-{cfg.network.matching}.h5')
    empty_dir = osp.join(outputs_dir, 'sfm_empty')
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')
    ba_dir = osp.join(deep_sfm_dir, 'ba_model')

    if cfg.redo:
        os.system(f'rm -rf {outputs_dir}')
        Path(outputs_dir).mkdir(exist_ok=True, parents=True)

        extract_features.main(img_lists, feature_out, cfg)
        if covis_num == -1:
            logger.warning(f"Exhaustive match all images")
            pairs_exhaustive_all.exhaustive_all_pairs(img_lists, covis_pairs_out)
        else:
            pairs_from_poses_hloc.covis_from_pose(img_lists, covis_pairs_out, covis_num, do_ba=True)
        match_features.main(cfg, feature_out, covis_pairs_out, matches_out, vis_match=False)
        generate_empty.generate_model(img_lists, empty_dir, do_ba=True)
        triangulation.main(deep_sfm_dir, empty_dir, outputs_dir, covis_pairs_out, feature_out,
                           matches_out, image_dir=None)

        global_ba.main(deep_sfm_dir, ba_dir)
        if cfg.parse_align_pose:
            parse_align_pose(ba_dir)
        else:
            logger.warning(f"Results are not parsed to data dir!")

@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()