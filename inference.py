from typing import ChainMap
import ray
import torch
import hydra
from tqdm import tqdm
import os
import os.path as osp
from pathlib import Path
import numpy as np
from loguru import logger
import math

from omegaconf.dictconfig import DictConfig

from src.inference.inference_OnePosePlus import inference_onepose_plus
from src.utils.ray_utils import ProgressBar, chunks

import json
import glob
import cv2
from PIL import Image

def get_img_full_path(img_path):
    return img_path.replace('/color/', '/color_full/')

def get_gt_pose_path(img_path):
    return img_path.replace('/color/', '/poses/').replace('.png', '.txt')

def get_intrin_path(img_path):
    return img_path.replace('/color/', '/intrin/').replace('.png', '.txt')

def get_3d_box_path(data_dir):
    refined_box_path = osp.join(data_dir, 'RefinedBox.txt')
    box_path = refined_box_path if osp.isfile(refined_box_path) else osp.join(data_dir, 'Box.txt')
    return box_path

def get_3d_anno(anno_3d_path):
    """ Read 3d information about this seq """
    with open(anno_3d_path, 'r') as f:
        anno_3d = json.load(f)
    
    descriptors3d = torch.Tensor(anno_3d['descriptors3d'])[None].cuda()
    keypoints3d = torch.Tensor(anno_3d['keypoints3d'])[None].cuda()
    scores3d = torch.Tensor(anno_3d['scores3d'])[None].cuda()
    anno_3d = {
        'keypoints3d': keypoints3d,
        'descriptors3d': descriptors3d,
        'scores3d': scores3d
    }
    return anno_3d


def get_default_paths(cfg):
    data_dir = cfg.input.data_dir
    sfm_model_dir = cfg.input.sfm_model_dir
    anno_dir = osp.join(sfm_model_dir, f'outputs_{cfg.match_type}_{cfg.network.detection}_{cfg.network.matching}/anno')
    anno_3d_path = osp.join(anno_dir, f'anno_3d_{cfg.feature_method}.json')
    
    img_lists = []
    color_dir = osp.join(data_dir, 'color')
    img_lists += glob.glob(color_dir + '/*.png', recursive=True)

    intrin_full_path = osp.join(data_dir, 'intrinsics.txt')
    paths = {
        'data_dir': data_dir,
        'sfm_model_dir': sfm_model_dir,
        'anno_3d_path': anno_3d_path,
        'intrin_full_path': intrin_full_path
    }
    return img_lists, paths

def vis_reproj(paths, img_path, pose_pred, pose_gt):
    """ Draw 2d box reprojected by 3d box"""
    from src.utils.objScanner_utils import parse_3d_box, parse_K
    from src.utils.vis_utils import reproj, draw_3d_box

    box_3d_path = get_3d_box_path(paths['data_dir'])
    box_3d, box3d_homo = parse_3d_box(box_3d_path)

    intrin_full_path = paths['intrin_full_path']
    K_full, K_full_homo = parse_K(intrin_full_path)

    image_full_path = get_img_full_path(img_path)
    image_full = cv2.imread(image_full_path)

    reproj_box_2d_gt = reproj(K_full, pose_gt, box_3d)
    draw_3d_box(image_full, reproj_box_2d_gt, color='y')
    if pose_pred is not None:
        reproj_box_2d_pred = reproj(K_full, pose_pred, box_3d)
        draw_3d_box(image_full, reproj_box_2d_pred, color='g')

    return image_full


def dump_vis3d(idx, cfg, image0, image1, image_full,
               kpts2d, kpts2d_reproj, confidence, inliers):
    """ Visualize by vis3d """
    from vis3d import Vis3D
    
    seq_name = '_'.join(cfg.input.data_dir.split('/')[-2:])
    if cfg.suffix:
        seq_name += '_' + cfg.suffix
    vis3d = Vis3D(cfg.output.vis_dir, seq_name)
    vis3d.set_scene_id(idx)

    # property for vis3d
    reproj_distance = np.linalg.norm(kpts2d_reproj - kpts2d, axis=1)
    inliers_bool = np.zeros((kpts2d.shape[0], 1), dtype=np.bool)
    if inliers is not None:
        inliers_bool[inliers] = True
        num_inliers = len(inliers)
    else:
        num_inliers = 0
    
    vis3d.add_keypoint_correspondences(image0, image1, kpts2d, kpts2d_reproj,
                                       metrics={
                                           'mconf': confidence.tolist(),
                                           'reproj_distance': reproj_distance.tolist()
                                       },
                                       booleans={
                                           'inliers': inliers_bool.tolist()
                                           
                                       },
                                       meta={
                                           'num_inliers': num_inliers,
                                           'width': image0.size[0],
                                           'height': image0.size[1],
                                       },
                                       name='matches')  
    image_full_pil = Image.fromarray(cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB))
    vis3d.add_image(image_full_pil, name='results')

@torch.no_grad()
def inference(cfg):
    # Load all test objects
    data_dirs = cfg.data_dir

    if isinstance(data_dirs, str):
        # Parse object directory
        num_val_seq = cfg.num_val_seq
        exception_obj_name_list = cfg.exception_obj_names
        top_k_obj = cfg.top_k_obj
        logger.info(
            f"Process all objects in directory:{data_dirs}, process: {num_val_seq if num_val_seq is not None else 'all'} sequences"
        )
        if num_val_seq is not None:
            assert num_val_seq != 0
            num_val_seq = -1 * num_val_seq
        
        if "want_seq_id" in cfg:
            num_val_seq = 0
            want_seq_id = cfg.want_seq_id
        else:
            want_seq_id = None

        object_names = os.listdir(data_dirs)[top_k_obj :]
        data_dirs_list = []

        if cfg.ids is not None:
            # Use data ids
            id2full_name = {name[:4]: name for name in object_names if "-" in name}
            object_names = [id2full_name[id] for id in cfg.ids if id in id2full_name]

        for object_name in object_names:
            if "-" not in object_name:
                continue

            if object_name in exception_obj_name_list:
                continue
            sequence_names = sorted(os.listdir(osp.join(data_dirs, object_name)))
            sequence_names = [
                sequence_name
                for sequence_name in sequence_names
                if ("-" in sequence_name) and ('-demo' not in sequence_name)
            ][num_val_seq:]

            obj_short_name = object_name.split('-', 2)[1]
            sequence_ids = [
                sequence_name.split('-',1)[1]
                for sequence_name in sequence_names
                if "-" in sequence_name
            ][num_val_seq:]

            if want_seq_id is not None:
                assert str(want_seq_id) in sequence_ids
                sequence_names = ['-'.join([obj_short_name, str(want_seq_id)])]

            print(sequence_names)
            data_dirs_list.append(
                " ".join([osp.join(data_dirs, object_name)] + sequence_names)
            )
    else:
        raise NotImplementedError

    data_dirs = data_dirs_list  # [obj_name]

    if not cfg.use_global_ray:
        name2metrics = inference_worker(data_dirs, cfg)
    else:
        # Init ray
        if cfg.ray.slurm:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg.ray.n_workers * cfg.ray.n_cpus_per_worker),
                num_gpus=math.ceil(cfg.ray.n_workers * cfg.ray.n_gpus_per_worker),
                local_mode=cfg.ray.local_mode,
                ignore_reinit_error=True,
            )
        logger.info(f"Use ray for inference, total: {cfg.ray.n_workers} workers")

        pb = ProgressBar(len(data_dirs), "Inference begin...")
        all_subsets = chunks(data_dirs, math.ceil(len(data_dirs) / cfg.ray.n_workers))
        sfm_worker_results = [
            inference_worker_ray_wrapper.remote(subset_data_dirs, cfg, pba=pb.actor, worker_id=id)
            for id, subset_data_dirs in enumerate(all_subsets)
        ]
        pb.print_until_done()
        results = ray.get(sfm_worker_results)
        name2metrics = dict(ChainMap(*results))
    
    # Parse metrics:
    gathered_metrics = {}
    for name, metrics in name2metrics.items():
        for metric_name, metric in metrics.items():
            if metric_name not in gathered_metrics:
                gathered_metrics[metric_name] = [metric]
            else:
                gathered_metrics[metric_name].append(metric)
        
    # Dump metrics:
    os.makedirs(cfg.output.txt_dir, exist_ok=True)
    with open(osp.join(cfg.output.txt_dir, 'metrics.txt'), 'w') as f:
        for name, metrics in name2metrics.items():
            f.write(f'{name}: \n')
            for metric_name, metric in metrics.items():
                f.write(f"{metric_name}: {metric}  ")
            f.write('\n ---------------- \n')
    
    with open(osp.join(cfg.output.txt_dir, 'metrics.txt'), 'a') as f:
        for metric_name, metric in gathered_metrics.items():
            print(f'{metric_name}:')
            metric_np = np.array(metric)
            metric_mean = np.mean(metric)
            print(metric_mean)
            print('---------------------')

            f.write('Summary: \n')
            f.writelines(str(metric_mean))
        
def inference_worker(data_dirs, cfg, pba=None, worker_id=0):
    logger.info(
        f"Worker {worker_id} will process: {[(data_dir.split(' ')[0]).split('/')[-1][:4] for data_dir in data_dirs]}, total: {len(data_dirs)} objects"
    )
    data_dirs = tqdm(data_dirs) if pba is None else data_dirs

    obj_name2metrics = {}
    for data_dir in data_dirs:
        logger.info(f"Processing {data_dir}.")

        # Load obj name and inference sequences
        root_dir, sub_dirs = data_dir.split(" ")[0], data_dir.split(" ")[1:]
        sfm_mapping_sub_dir = '-'.join([sub_dirs[0].split("-")[0], '1'])
        num_img_in_mapping_seq = len(os.listdir(osp.join(root_dir, sfm_mapping_sub_dir, 'color')))
        obj_name = root_dir.split("/")[-1]
        sfm_base_path = cfg.sfm_base_dir

        if "object_detector_method" in cfg:
            object_detector_method = cfg.object_detector_method
        else:
            object_detector_method = 'GT'

        # Get all inference image path
        all_image_paths = []
        for sub_dir in sub_dirs:

            if object_detector_method == 'GT':
                color_dir = osp.join(root_dir, sub_dir, "color")
            else:
                raise NotImplementedError

            img_paths = list(Path(color_dir).glob("*.png"))
            if len(img_paths) == num_img_in_mapping_seq:
                logger.warning(f"Same num of images in test sequence:{sub_dir}")
            image_paths = [str(img_path) for img_path in img_paths]
            all_image_paths += image_paths

        if len(all_image_paths) == 0:
            logger.info(f"No png image in {root_dir}")
            if pba is not None:
                pba.update.remote(1)
            continue

        sfm_results_dir = osp.join(
            sfm_base_path,
            "outputs_"
            + cfg.match_type
            + "_"
            + cfg.network.detection
            + "_"
            + cfg.network.matching,
            obj_name,
        )

        metrics = inference_onepose_plus(sfm_results_dir, all_image_paths, cfg, use_ray=cfg.use_local_ray, verbose=cfg.verbose)
        obj_name2metrics[obj_name] = metrics
        if pba is not None:
            pba.update.remote(1)
    
    return obj_name2metrics

@ray.remote(num_cpus=1)
def inference_worker_ray_wrapper(*args, **kwargs):
    return inference_worker(*args, **kwargs)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
