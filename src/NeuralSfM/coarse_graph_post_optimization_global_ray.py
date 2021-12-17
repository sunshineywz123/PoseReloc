import argparse
from collections import ChainMap
import os
import os.path as osp
import copy

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
import ray
import math
import random

from src.utils.ray_utils import ProgressBar
from src.post_optimization.post_optimization import post_optimization_ray_warp
from src.post_optimization.utils.ray_utils import chunks
from src.post_optimization.utils.eval_metric_utils import imc_bag_pose_auc


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Match and refinement model related parameters
    # TODO: move configs to cfg
    parser.add_argument("--cfg_path", type=str, default="")
    parser.add_argument("--weight_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=666)

    # Data related
    # NOTE: need to be identity to coarse graph creator
    parser.add_argument("--img_resize_max", type=int, default=1920)
    parser.add_argument("--img_resize_min", type=int, default=800)
    parser.add_argument("--df", type=int, default=8, help="divisible factor.")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="IMC",
        choices=["IMC"],
        help="Choice: [IMC,]",
    )
    parser.add_argument(
        "--n_sub_bag",
        type=int,
        default=None,
        help="Only process small subset bag, for debug!",
    )
    parser.add_argument(
        "--data_part_name",
        default=None,
        help="Used to identify different dataset results",
    )
    parser.add_argument("--n_imgs", default=None, help="Used for debug")

    # COLMAP results related:
    parser.add_argument(
        "--colmap_results_load_dir", default=None, help="Get colmap output results"
    )
    parser.add_argument(
        "--colmap_refined_save_dir",
        default=None,
        help="Colmap refined results save dir",
    )
    parser.add_argument("--colmap_best_index", type=int, default=0)

    # Ray related
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--n_cpus_per_worker", type=float, default=1)
    parser.add_argument("--n_gpus_per_worker", type=float, default=1)
    parser.add_argument(
        "--local_mode", action="store_true", help="ray local mode for debugging."
    )

    # Others
    parser.add_argument("--fine_match_debug", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--evaluation", action="store_true")

    args = parser.parse_args()

    # Post process of args
    args.match_type = "loftr_coarse"

    return args


def main():
    args = parse_args()

    if args.slurm:
        ray.init(address=os.environ["ip_head"])
    else:
        ray.init(
            num_cpus=math.ceil(args.n_workers * args.n_cpus_per_worker),
            num_gpus=math.ceil(args.n_workers * args.n_gpus_per_worker),
            local_mode=args.local_mode,
        )

    sub_set = args.n_sub_bag  # For debug

    # Get scene subset bag information
    data_root = args.data_root
    subset_path = osp.join(data_root, "sub_set")
    subset_list = sorted(os.listdir(subset_path))[:sub_set]

    random.shuffle(subset_list)

    pb = ProgressBar(len(subset_list), "Run pose post optimization")
    all_subsets = chunks(subset_list, math.ceil(len(subset_list) / args.n_workers))
    copyed_args = [copy.deepcopy(args) for i in range(args.n_workers)]
    obj_refs = [
        post_optimization_ray_warp.remote(subsets, copyed_args[i], pb.actor)
        for i, subsets in enumerate(all_subsets)
    ]
    pb.print_until_done()
    results = ray.get(obj_refs)
    pose_error_before_refine = dict(ChainMap(*[k for k, _ in results]))
    pose_error_after_refine = dict(ChainMap(*[s for _, s in results]))

    # Eval aggregated results
    scene_total_save_dir = osp.join(
        args.colmap_refined_save_dir, args.match_type, args.data_part_name
    )

    # Calculate per-scene pose auc
    imc_bag_pose_auc(
        pose_error_before_refine,
        save_dir=scene_total_save_dir,
        base_save_name="before_refine_scene_pose_auc.txt",
    )
    imc_bag_pose_auc(
        pose_error_after_refine,
        save_dir=scene_total_save_dir,
        base_save_name="after_refine_scene_pose_auc.txt",
    )


if __name__ == "__main__":
    main()
