from itertools import chain
import ray
import os
import math
from tqdm import tqdm
import numpy as np
from loguru import logger
import torch
import os.path as osp
from time import time

from src.datasets.GATs_loftr_inference_dataset_for_demo import GATs_loftr_inference_dataset
from src.utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict
from src.architectures.GATs_LoFTR.GATs_LoFTR import GATs_LoFTR
from src.utils.metric_utils import aggregate_metrics
from src.utils.visualize.dump_vis3d import dump_obj, dump_obj_with_feature_map
from src.utils.metric_utils import compute_query_pose_errors
from src.utils.vis_utils import vis_pca_features

from .inference_gats_loftr_worker import (
    inference_gats_loftr_worker,
    inference_gats_loftr_worker_ray_wrapper,
)


def build_model(model_configs, ckpt_path):
    match_model = GATs_LoFTR(model_configs)

    # load checkpoints
    logger.info(f"Load ckpt:{ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k.replace("matcher.", "")] = state_dict.pop(k)

    match_model.load_state_dict(state_dict, strict=True)
    match_model.eval()
    return match_model


def inference_gats_loftr_vis(
    sfm_results_dir, all_image_paths, cfg, use_ray=True, verbose=True, vis3d_pth=None
):
    """
    Inference for one object
    """

    # Build dataset:
    dataset = GATs_loftr_inference_dataset(
        sfm_results_dir,
        all_image_paths,
        load_3d_coarse=cfg.datamodule.load_3d_coarse,
        shape3d=cfg.datamodule.shape3d_val,
        num_leaf=cfg.datamodule.num_leaf,
        img_pad=cfg.datamodule.img_pad,
        img_resize=cfg.datamodule.img_resize,
        df=cfg.datamodule.df,
        pad=cfg.datamodule.pad3D,
        load_pose_gt=True,
        n_images=None,
    )
    match_model = build_model(cfg["model"]["loftr"], cfg["model"]["pretrained_ckpt"])

    match_model.cuda()
    results = []

    for data in tqdm(dataset):
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }

        result = extract_matches(
            data_c, match_model, metrics_configs=cfg["model"]["eval_metrics"]
        )

        results += [result]

    logger.info("Match and compute pose error finish!")

    # Parse results:
    R_errs = []
    t_errs = []
    time = []
    if "ADD_metric" in results[0]:
        add_metric = []
        proj2d_metric = []
    else:
        add_metric = None
        proj2d_metric = None

    # Gather results metrics:
    for result in results:
        R_errs.append(result["R_errs"])
        t_errs.append(result["t_errs"])
        time.append(result["time"])
        if add_metric is not None:
            add_metric.append(result["ADD_metric"])
            proj2d_metric.append(result["proj2D_metric"])

    # Write results to vis3d
    if vis3d_pth is not None:
        vis3d_dir, name = vis3d_pth.rsplit("/", 1)
        dump_obj_with_feature_map(results, vis3d_dir, name)

    # Aggregate metrics:
    pose_errs = {"R_errs": R_errs, "t_errs": t_errs}
    if add_metric is not None:
        pose_errs.update({"ADD_metric": add_metric, "proj2D_metric": proj2d_metric})
    metrics = aggregate_metrics(
        pose_errs, cfg["model"]["eval_metrics"]["pose_thresholds"]
    )
    metrics.update({"time": np.mean(time)})

    # TODO: add visualize
    return metrics


@torch.no_grad()
def extract_matches(data, match_model, metrics_configs):
    # 1. Run inference
    start_time = time()
    match_model(data, return_fine_unfold_feat=True, return_coarse_atten_feat=True)
    end_time = time()
    # logger.info(f"consume: {end_time - start_time}")

    # 2. Compute metrics
    compute_query_pose_errors(data, metrics_configs)

    # 3. Vis pca feature
    pca_features_dict = vis_pca_features(data)

    R_errs = data["R_errs"]
    t_errs = data["t_errs"]
    inliers = data["inliers"]
    pose_pred = [data["pose_pred"][0]]

    result_data = {
        "mkpts3d": data["mkpts_3d_db"].cpu().numpy(),
        "mkpts_query": data["mkpts_query_f"].cpu().numpy(),
        "mconf": data["mconf"].cpu().numpy(),
        "R_errs": R_errs,
        "t_errs": t_errs,
        "inliers": inliers,
        "pose_pred": pose_pred,
        "pose_gt": data["query_pose_gt"][0].cpu().numpy(),
        "intrinsic": data["query_intrinsic"][0].cpu().numpy(),
        "image_path": data["query_image_path"],
        "time": end_time - start_time,
    }

    if "ADD" in data:
        result_data.update({"ADD_metric": data["ADD"]})
        result_data.update({"proj2D_metric": data["proj2D"]})
    
    result_data.update(pca_features_dict)

    del data
    torch.cuda.empty_cache()

    return result_data
