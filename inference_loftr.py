from typing import ChainMap
import ray
import torch
import hydra
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
from loguru import logger
import math
import pandas as pd

from omegaconf.dictconfig import DictConfig

from src.inference.inference_gats_loftr.inference_gats_loftr import inference_gats_loftr
from src.utils.ray_utils import ProgressBar, chunks


@torch.no_grad()
def inference(cfg):
    # Load all test objects
    data_dirs = cfg.data_dir

    if isinstance(data_dirs, str):
        # Parse object directory
        # assert isinstance(data_dirs, str)
        num_val_seq = cfg.num_val_seq
        exception_obj_name_list = cfg.exception_obj_names
        top_k_obj = cfg.top_k_obj
        logger.info(
            f"Process all objects in directory:{data_dirs}, process: {num_val_seq if num_val_seq is not None else 'all'} sequences"
        )
        if num_val_seq is not None:
            assert num_val_seq > 0
            num_val_seq = -1 * num_val_seq

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
                if "-" in sequence_name
            ][num_val_seq:]
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
            for id,subset_data_dirs in enumerate(all_subsets)
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
    name2metrics_sorted = {k:v for k,v in sorted(name2metrics.items(), key= lambda item: item[1]['5cm@5degree'])}
    os.makedirs(cfg.output.txt_dir, exist_ok=True)
    with open(osp.join(cfg.output.txt_dir, 'metrics.txt'), 'w') as f:
        for name, metrics in name2metrics_sorted.items():
            f.write(f'{name}: \n')
            for metric_name, metric in metrics.items():
                f.write(f"{metric_name}: {metric}  ")
            f.write('\n ---------------- \n')
    
    with open(osp.join(cfg.output.txt_dir, 'metrics.txt'), 'a') as f:
        for metric_name, metric in gathered_metrics.items():
            print(f'{metric_name}:')
            metric_parsed = pd.DataFrame(metric)
            print(metric_parsed.describe())
            print('---------------------')

            # f.write('Summary: \n')
            # f.writelines(metric_parsed.describe())
        
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
        obj_name = root_dir.split("/")[-1]
        sfm_base_path = cfg.sfm_base_dir

        # Get all inference image path
        all_image_paths = []
        for sub_dir in sub_dirs:
            color_dir = osp.join(root_dir, sub_dir, "color")
            img_names = os.listdir(color_dir)
            image_paths = [osp.join(color_dir, img_name) for img_name in img_names]
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

        if cfg.output.visual_vis3d:
            os.makedirs(cfg.output.vis_dir, exist_ok=True)
            vis3d_pth = osp.join(cfg.output.vis_dir, obj_name)
        else:
            None

        metrics = inference_gats_loftr(sfm_results_dir, all_image_paths, cfg, use_ray=cfg.use_local_ray, verbose=cfg.verbose, vis3d_pth=vis3d_pth)
        obj_name2metrics[obj_name] = metrics
        if pba is not None:
            pba.update.remote(1)
    
    return obj_name2metrics



@ray.remote
def inference_worker_ray_wrapper(*args, **kwargs):
    return inference_worker(*args, **kwargs)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
