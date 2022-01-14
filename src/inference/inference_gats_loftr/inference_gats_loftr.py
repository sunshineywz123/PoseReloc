from itertools import chain
import ray
import os
import math
import numpy as np
from loguru import logger
import torch

from src.datasets.GATs_loftr_inference_dataset import GATs_loftr_inference_dataset
from src.utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict
from src.architectures.GATs_LoFTR.GATs_LoFTR import GATs_LoFTR
from .inference_gats_loftr_worker import (
    inference_gats_loftr_worker,
    inference_gats_loftr_worker_ray_wrapper,
)
from src.utils.metric_utils import aggregate_metrics

args = {
    "ray": {
        "slurm": False,
        "n_workers": 4,
        # "n_cpus_per_worker": 1,
        "n_cpus_per_worker": 1,
        "n_gpus_per_worker": 0.25,
        "local_mode": False,
    },
}

def build_model(model_configs, ckpt_path):
    match_model = GATs_LoFTR(model_configs)
    # load checkpoints
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k.replace("matcher.", "")] = state_dict.pop(k)

    match_model.load_state_dict(state_dict, strict=True)
    match_model.eval()
    return match_model

def inference_gats_loftr(
    sfm_results_dir, all_image_paths, cfg, use_ray=True, verbose=True
):
    """
    Inference for one object
    """

    # Build dataset:
    dataset = GATs_loftr_inference_dataset(
        sfm_results_dir,
        all_image_paths,
        shape3d=cfg.datamodule.shape3d_val,
        num_leaf=cfg.datamodule.num_leaf,
        img_pad=cfg.datamodule.img_pad,
        img_resize=cfg.datamodule.img_resize,
        df=cfg.datamodule.df,
        pad=True,
        load_pose_gt=True,
    )
    match_model = build_model(cfg['model']["loftr"], cfg['model']['pretrained_ckpt'])

    # Run matching
    if use_ray:
        # Initial ray:
        cfg_ray = args["ray"]
        if cfg_ray["slurm"]:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                num_gpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_gpus_per_worker"]),
                local_mode=cfg_ray["local_mode"],
                ignore_reinit_error=True,
            )

        pb = (
            ProgressBar(len(dataset), "Matching image pairs...")
            if verbose
            else None
        )
        all_subset_ids = chunk_index(
            len(dataset), math.ceil(len(dataset) / cfg_ray["n_workers"])
        )
        all_subset_ids = all_subset_ids

        obj_refs = [
            inference_gats_loftr_worker_ray_wrapper.remote(
                dataset,
                match_model,
                subset_ids,
                cfg['model'],
                pb.actor if pb is not None else None,
                verbose=verbose,
            )
            for subset_ids in all_subset_ids
        ]
        pb.print_until_done() if pb is not None else None
        results = ray.get(obj_refs)

        R_errs = list(chain(* [k for k, _, _, _ in results]))
        t_errs = list(chain(* [k for _, k, _, _ in results]))
        inliers = list(chain(* [k for _, _, k, _ in results]))
        poses_pred = list(chain(* [k for _, _, _, k in results]))
        logger.info("Matcher finish!")
    else:
        all_ids = np.arange(0, len(dataset))
        R_errs, t_errs, inliers, pose_pred = inference_gats_loftr_worker(dataset, match_model, all_ids, cfg['model'], verbose=verbose)
        logger.info("Match and compute pose error finish!")

    # Aggregate metrics: 
    pose_errs = {'R_errs': R_errs, "t_errs": t_errs}
    metrics = aggregate_metrics(pose_errs, cfg['model']['eval_metrics']['pose_thresholds'])

    # TODO: add visualize
    return metrics