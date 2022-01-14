from loguru import logger
import ray
import torch
from tqdm import tqdm
from time import time
from src.utils.metric_utils import compute_query_pose_errors

@torch.no_grad()
def extract_matches(data, match_model, metrics_configs):
    # 1. Run inference
    start_time = time()
    match_model(data)
    end_time = time()
    # logger.info(f"consume: {end_time - start_time}")

    # 2. Compute metrics
    compute_query_pose_errors(data, metrics_configs)

    R_errs = data["R_errs"]
    t_errs = data["t_errs"]
    inliers = data["inliers"]
    pose_pred = [data["pose_pred"][0]]

    del data
    torch.cuda.empty_cache()

    return R_errs, t_errs, inliers, pose_pred


def inference_gats_loftr_worker(dataset, match_model, subset_ids, cfgs, pba=None, verbose=True):
    match_model.cuda()
    R_errs_all, t_errs_all, inliers_all, pose_pred_all = [], [], [], []

    if verbose:
        subset_ids = tqdm(subset_ids) if pba is None else subset_ids
    else:
        assert pba is None
        subset_ids = subset_ids

    for subset_id in subset_ids:
        data = dataset[subset_id]
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }

        R_errs, t_errs, inliers, pose_pred = extract_matches(
            data_c, match_model, metrics_configs=cfgs["eval_metrics"]
        )

        R_errs_all += R_errs
        t_errs_all += t_errs
        inliers_all += inliers
        pose_pred_all += pose_pred

        if pba is not None:
            pba.update.remote(1)

    return R_errs_all, t_errs_all, inliers_all, pose_pred_all


@ray.remote(num_cpus=1, num_gpus=0.25)  # release gpu after finishing
def inference_gats_loftr_worker_ray_wrapper(*args, **kwargs):
    return inference_gats_loftr_worker(*args, **kwargs)
