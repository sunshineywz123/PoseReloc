import json
import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
import glob
import hydra
import torch
import time

import open3d as o3d
import os.path as osp

from loguru import logger
from pathlib import Path
from matplotlib.pyplot import box
from omegaconf import DictConfig


def parseScanData(cfg):
    """ Parse arkit scanning data"""
    #TODO: add arkit data processing
    pass


def merge_(anno_2d_file, avg_anno_3d_file, collect_anno_3d_file,
           idxs_file, img_id, ann_id, images, annotations):
    """ Merge annotations about difference objs"""
    with open(anno_2d_file, 'r') as f:
        annos_2d = json.load(f)
    
    for anno_2d in annos_2d:
        img_id += 1
        info = {
            'id': img_id,
            'img_file': anno_2d['img_file'],
        }
        images.append(info)

        ann_id += 1
        anno = {
            'image_id': img_id,
            'id': ann_id,
            'pose_file': anno_2d['pose_file'],
            'anno2d_file': anno_2d['anno_file'],
            'avg_anno3d_file': avg_anno_3d_file,
            'collect_anno3d_file': collect_anno_3d_file,
            'idxs_file': idxs_file
        }
        annotations.append(anno)
    return img_id, ann_id


def merge_anno(cfg):
    """ Merge different objects' anno file into one anno file """
    anno_dirs = []
    
    for name in cfg.names:
        anno_dir = osp.join(cfg.datamodule.data_dir, name, f'outputs_{cfg.match_type}_{cfg.network.detection}_{cfg.network.matching}', 'anno')
        anno_dirs.append(anno_dir) 
    
    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    for anno_dir in anno_dirs:
        logger.info(f'Merging anno dir: {anno_dir}')
        anno_2d_file = osp.join(anno_dir, 'anno_2d.json')
        avg_anno_3d_file = osp.join(anno_dir, 'anno_3d_average.npz')
        collect_anno_3d_file = osp.join(anno_dir, 'anno_3d_collect.npz')
        idxs_file = osp.join(anno_dir, 'idxs.npy')

        if not osp.isfile(anno_2d_file) or not osp.isfile(avg_anno_3d_file) or not osp.isfile(collect_anno_3d_file):
            logger.info(f'No annotation in: {anno_dir}')
            continue
        
        img_id, ann_id = merge_(anno_2d_file, avg_anno_3d_file, collect_anno_3d_file,
                                idxs_file, img_id, ann_id, images, annotations)
    
    logger.info(f'Total num: {len(images)}')
    instance = {'images': images, 'annotations': annotations}

    out_dir = osp.dirname(cfg.datamodule.out_path)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with open(cfg.datamodule.out_path, 'w') as f:
        json.dump(instance, f)


def sfm(cfg):
    """ Sparse reconstruction and postprocess (on 3d points and features)"""
    data_dirs = cfg.dataset.data_dir
    
    for data_dir in data_dirs:
        logger.info(f"Processing {data_dir}.")
        root_dir, sub_dirs = data_dir.split(' ')[0], data_dir.split(' ')[1:]

        img_lists = []
        for sub_dir in sub_dirs:
            seq_dir = osp.join(root_dir, sub_dir)
            img_lists += glob.glob(str(Path(seq_dir)) + '/color/*.png', recursive=True)

        # ------------------ downsample ------------------
        down_img_lists = []
        down_ratio = 5
        # down_ratio = 20
        for img_file in img_lists:
            index = int(img_file.split('/')[-1].split('.')[0])
            if index % down_ratio == 0:
                down_img_lists.append(img_file)  
        img_lists = down_img_lists
        # -------------------------------------------------

        if len(img_lists) == 0:
            logger.info(f"No png image in {root_dir}")
            continue
        
        obj_name = root_dir.split('/')[-1]
        outputs_dir_root = cfg.dataset.outputs_dir.format(obj_name)

        sfm_core(cfg, img_lists, outputs_dir_root)
        postprocess(cfg, img_lists, root_dir, sub_dirs, outputs_dir_root) 


def sfm_core(cfg, img_lists, outputs_dir_root):
    """ Sparse reconstruction: extract features, match features, triangulation"""
    from src.hloc import extract_features, pairs_from_covisibility, match_features, \
                         generate_empty, triangulation, pairs_from_poses
    from src.NeuralSfM import coarse_match, post_optimization
    outputs_dir = osp.join(outputs_dir_root, 'outputs_' + cfg.match_type + '_' + cfg.network.detection + '_' + cfg.network.matching)
    vis3d_pth = osp.join(outputs_dir_root, 'vis3d', 'outputs_' + cfg.match_type + '_' + cfg.network.detection + '_' + cfg.network.matching)

    feature_out = osp.join(outputs_dir, f'feats-{cfg.network.detection}.h5')
    covis_num = cfg.sfm.covis_num
    covis_pairs_out = osp.join(outputs_dir, f'pairs-covis{covis_num}.txt')
    matches_out = osp.join(outputs_dir, f'matches-{cfg.network.matching}.h5')
    empty_dir = osp.join(outputs_dir, 'sfm_empty')
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')
    visualize_dir = osp.join(outputs_dir, 'visualize')
    
    if cfg.redo:
        if cfg.network.matching != 'loftr':
            os.system(f'rm -rf {outputs_dir}') 
            Path(outputs_dir).mkdir(exist_ok=True, parents=True)

            extract_features.main(img_lists, feature_out, cfg)
            # pairs_from_covisibility.covis_from_index(img_lists, covis_pairs_out, num_matched=covis, gap=cfg.sfm.gap)
            pairs_from_poses.covis_from_pose(img_lists, covis_pairs_out, covis_num, max_rotation=cfg.sfm.rotation_thresh)
            match_features.main(cfg, feature_out, covis_pairs_out, matches_out, vis_match=False)
            generate_empty.generate_model(img_lists, empty_dir)
            triangulation.main(deep_sfm_dir, empty_dir, outputs_dir, covis_pairs_out, feature_out, matches_out, image_dir=None)
            torch.cuda.synchronize()
        else:
            if cfg.overwrite_all:
                os.system(f'rm -rf {outputs_dir}') 
                os.system(f'rm -rf {vis3d_pth}') 
            Path(outputs_dir).mkdir(exist_ok=True, parents=True)

            if not osp.exists(osp.join(deep_sfm_dir, "model_coarse")) or cfg.overwrite_coarse:
                logger.info('LoFTR coarse mapping begin...')
                os.system(f'rm -rf {empty_dir}')
                os.system(f'rm -rf {deep_sfm_dir}')

                pairs_from_covisibility.covis_from_index(img_lists, covis_pairs_out, num_matched=covis_num, gap=cfg.sfm.gap)
                coarse_match.loftr_coarse_matching(img_lists, covis_pairs_out, feature_out, matches_out, use_ray=cfg.use_local_ray)
                generate_empty.generate_model(img_lists, empty_dir)
                triangulation.main(deep_sfm_dir, empty_dir, outputs_dir, covis_pairs_out, feature_out, matches_out, match_model=cfg.network.matching, image_dir=None)

                if cfg.enable_loftr_post_refine:
                    assert osp.exists(osp.join(deep_sfm_dir, 'model'))
                    os.system(f"mv {osp.join(deep_sfm_dir, 'model')} {osp.join(deep_sfm_dir, 'model_coarse')}")
                    
            if cfg.enable_loftr_post_refine:
                if not osp.exists(osp.join(deep_sfm_dir, 'model')) or cfg.overwrite_fine:
                    assert osp.exists(osp.join(deep_sfm_dir, 'model_coarse')), f'model_coarse not exist under: {deep_sfm_dir}, please set \'cfg.overwrite_coarse = True\''
                    os.system(f"rm -rf {osp.join(deep_sfm_dir, 'model')}")

                    logger.info('LoFTR mapping post refinement begin...')
                    state, _, _ = post_optimization.post_optimization(img_lists, covis_pairs_out, colmap_coarse_dir=osp.join(deep_sfm_dir, 'model_coarse'), refined_model_save_dir= osp.join(deep_sfm_dir, 'model'), fine_match_use_ray=cfg.use_local_ray, visualize_dir=visualize_dir, vis3d_pth=vis3d_pth)
                    if state == False:
                        logger.error('colmap coarse is empty!')
    
    
def postprocess(cfg, img_lists, root_dir, sub_dirs, outputs_dir_root):
    """ Filter points and average feature"""
    from src.utils.colmap.read_write_model import read_model
    from src.hloc.postprocess import filter_points, feature_process, filter_tkl

    data_dir0 = osp.join(root_dir, sub_dirs[0])
    # bbox_path = osp.join(data_dir0, 'RefinedBox.txt')
    # bbox_path = bbox_path if osp.isfile(bbox_path) else osp.join(data_dir0, 'Box.txt')
    bbox_path = osp.join(data_dir0, 'Box.txt')
    trans_box_path = osp.join(data_dir0, 'Box_trans.txt')

    match_type = cfg.match_type
    outputs_dir = osp.join(outputs_dir_root, 'outputs_' + cfg.match_type + '_' + cfg.network.detection + '_' + cfg.network.matching)
    feature_out = osp.join(outputs_dir, f'feats-{cfg.network.detection}.h5')
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')
    model_path = osp.join(deep_sfm_dir, 'model')

    # select track length to limit the number of 3d points below thres.
    track_length, points_count_list = filter_tkl.get_tkl(model_path, thres=cfg.dataset.max_num_kp3d, show=False) 
    tkl_file_path = filter_tkl.vis_tkl_filtered_pcds(model_path, points_count_list, track_length, outputs_dir) # visualization only

    xyzs, points_idxs = filter_points.filter_3d(model_path, track_length, bbox_path, box_trans_path=trans_box_path) # crop 3d points by 3d box and track length
    merge_xyzs, merge_idxs = filter_points.merge(xyzs, points_idxs, dist_threshold=1e-3) # merge 3d points by distance between points
    if cfg.debug:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merge_xyzs)
        out_file = osp.join(outputs_dir, 'box_filter.ply') 
        o3d.io.write_point_cloud(out_file, pcd)

    # FIXME: no param detector and debug
    feature_process.get_kpt_ann(cfg, img_lists, feature_out, outputs_dir, merge_idxs, merge_xyzs)
    

@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()