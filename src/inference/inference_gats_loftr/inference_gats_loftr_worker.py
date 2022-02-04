from unittest import result
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

    result_data = {
        "mkpts3d": data['mkpts_3d_db'].cpu().numpy(),
        "mkpts_query": data['mkpts_query_f'].cpu().numpy(),
        "mconf": data['mconf'].cpu().numpy(),
        "R_errs": R_errs,
        "t_errs": t_errs,
        "inliers": inliers,
        "pose_pred": pose_pred,
        "pose_gt": data['query_pose_gt'][0].cpu().numpy(),
        "intrinsic": data['query_intrinsic'][0].cpu().numpy(),
        'image_path': data['query_image_path']
    }

    del data
    torch.cuda.empty_cache()

    return result_data


def inference_gats_loftr_worker(
    dataset, match_model, subset_ids, cfgs, pba=None, verbose=True
):
    match_model.cuda()
    results = []

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

        result = extract_matches(
            data_c, match_model, metrics_configs=cfgs["eval_metrics"]
        )

        results += [result]

        if pba is not None:
            pba.update.remote(1)

    return results


@ray.remote(num_cpus=1, num_gpus=0.5)  # release gpu after finishing
def inference_gats_loftr_worker_ray_wrapper(*args, **kwargs):
    return inference_gats_loftr_worker(*args, **kwargs)
