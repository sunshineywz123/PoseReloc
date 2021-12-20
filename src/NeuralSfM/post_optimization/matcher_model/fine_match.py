import os
import math
from loguru import logger
from typing import ChainMap
from src.utils.ray_utils import ProgressBar, chunk_index
from .fine_match_worker import *


def fine_matcher(cfgs, matching_pairs_dataset, visualize_dir=None, use_ray=False):
    detector, matcher = build_model(cfgs["model"])

    if not use_ray:
        subset_ids = range(len(matching_pairs_dataset))
        fine_match_results = matchWorker(
            matching_pairs_dataset,
            subset_ids,
            detector,
            matcher,
            visualize=cfgs["visualize"],
            visualize_dir=visualize_dir,
        )
    else:
        # Initial ray:
        cfg_ray = cfgs["ray"]
        if cfg_ray["slurm"]:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                num_gpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_gpus_per_worker"]),
                local_mode=cfg_ray["local_mode"], ignore_reinit_error=True
            )

        pb = ProgressBar(len(matching_pairs_dataset), "Matching image pairs...")
        all_subset_ids = chunk_index(
            len(matching_pairs_dataset),
            math.ceil(len(matching_pairs_dataset) / cfg_ray["n_workers"]),
        )
        obj_refs = [
            matchWorker_ray_wrapper.remote(
                matching_pairs_dataset,
                subset_ids,
                detector,
                matcher,
                visualize=cfgs["visualize"],
                visualize_dir=visualize_dir,
                pba=pb.actor,
            )
            for subset_ids in all_subset_ids
        ]
        pb.print_until_done()
        results = ray.get(obj_refs)
        fine_match_results = dict(ChainMap(*results))

    logger.info("Matcher finish!")
    return fine_match_results
