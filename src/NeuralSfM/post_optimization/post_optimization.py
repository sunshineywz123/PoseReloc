import os
import os.path as osp
import copy

from ray.actor import ActorHandle

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
from loguru import logger
import ray
from src.utils.data_io import load_obj, save_obj

from src.datasets.neuralsfm_coarse_colmap_dataset import CoarseColmapDataset
from .data_construct import (
    MatchingPairData,
    ConstructOptimizationData,
)
from .matcher_model import *
from .optimizer.optimizer import Optimizer

cfgs = {
    "coarse_colmap_data": {
        "img_resize": 512,
        "df": 8,
        "feature_track_assignment_strategy": "greedy",
        # "feature_track_assignment_strategy": "balance",
        "verbose": True,
    },
    "fine_match_debug": True,
    "fine_matcher": {
        "model": {
            "cfg_path": "configs/loftr_configs/loftr_w9_no_cat_coarse.py",
            "weight_path": "weight/loftr_w9_no_cat_coarse_auc10=0.685.ckpt",
            "seed": 666,
        },
        "visualize": False,  # Visualize fine feature map and corresponds
        "ray": {
            "slurm": False,
            "n_workers": 4,
            "n_cpus_per_worker": 1,
            "n_gpus_per_worker": 0.25,
            "local_mode": False,
        },
    },
    "optimizer": {
        # Dataloading related:
        "num_workers": 12,
        "batch_size": 2000,
        "solver_type": "FirstOrder",
        "residual_mode": "feature_metric_error",  # ["feature_metric_error", "geometry_error"]
        # "residual_mode": "geometry_error",  # ["feature_metric_error", "geometry_error"]
        "distance_loss_scale": 10,  # only available for featuremetric error mode
        "optimize_lr": {
            "depth": 1e-2,
            "pose": 1e-5,
            "BA": 1e-5,
        },  # Only available for first order solver
        # "optim_procedure": ["depth", 'pose'] * 3 + ['BA'],
        "optim_procedure": ["depth"] * 3,
        "image_i_f_scale": 2,  # For Loftr is 2, don't change!
        "verbose": True,
    },
    "visualize": True,  # vis3d visualize
    "evaluation": False,
}


def post_optimization(
    image_lists,
    covis_pairs_pth,
    colmap_coarse_dir,
    refined_model_save_dir,
    fine_match_use_ray=False,  # Use ray for fine match
    visualize_dir=None,
    vis3d_pth=None,
):
    # Construct scene data
    colmap_image_dataset = CoarseColmapDataset(
        cfgs["coarse_colmap_data"],
        image_lists,
        covis_pairs_pth,
        colmap_coarse_dir,
        refined_model_save_dir,
        vis_path=vis3d_pth if vis3d_pth is not None else None,
    )
    logger.info("Scene data construct finish!")

    state = colmap_image_dataset.state
    if state == False:
        logger.warning(
            f"Build colmap coarse dataset fail! colmap point3D or images or cameras is empty!"
        )
        return state, None, None

    # Construct matching data
    matching_pairs_dataset = MatchingPairData(colmap_image_dataset)

    # Fine level match
    save_path = osp.join(covis_pairs_pth.rsplit("/", 1)[0], "fine_matches.pkl")
    if not osp.exists(save_path) or cfgs["fine_match_debug"]:
        logger.info(f"Fine matching begin!")
        fine_match_results = fine_matcher(
            cfgs["fine_matcher"],
            matching_pairs_dataset,
            visualize_dir,
            use_ray=fine_match_use_ray,
        )
        save_obj(fine_match_results, save_path)
    else:
        logger.info(f"Fine matches exists! Load from {save_path}")
        fine_match_results = load_obj(save_path)

    # Construct depth optimization data
    optimization_data = ConstructOptimizationData(
        colmap_image_dataset, fine_match_results
    )

    # Post optimization
    optimizer = Optimizer(optimization_data, cfgs["optimizer"])
    results_dict = optimizer()

    # Update results
    (
        pose_error_before_refine,
        pose_error_after_refine,
    ) = colmap_image_dataset.update_optimize_results_to_colmap(
        results_dict, visualize=cfgs["visualize"], evaluation=cfgs["evaluation"]
    )

    return state, pose_error_before_refine, pose_error_after_refine


@ray.remote(num_cpus=1, num_gpus=1, max_calls=1)  # release gpu after finishing
def post_optimization_ray_warp(subset_bag, args, pba: ActorHandle):
    error_before_refine = {}
    error_after_refine = {}
    args_reserve = copy.deepcopy(args)
    for subset_name in subset_bag:
        logger.info(f"Post optimize bag: {subset_name}")
        args = copy.deepcopy(args_reserve)

        args.subset_name = subset_name
        # Get colmap results path and raw match path
        base_dir_part = [args.colmap_results_load_dir]
        base_dir_part.append(args.match_type)
        base_dir_part.append(
            args.data_part_name
        ) if args.data_part_name is not None else None
        base_dir_part.append(osp.splitext(subset_name)[0])
        results_load_dir_medium = osp.join(*base_dir_part)
        args.scene_results_base_dir = results_load_dir_medium
        args.colmap_results_load_dir = osp.join(
            results_load_dir_medium, "colmap_output_path", str(args.colmap_best_index)
        )
        args.raw_match_results_dir = osp.join(
            results_load_dir_medium, "raw_matches/raw_matches.h5"
        )

        if not osp.exists(args.colmap_results_load_dir):
            logger.warning(
                f"Colmap load path:{args.colmap_results_load_dir} not exists!"
            )
            pba.update.remote(1)
            continue

        # Get save path
        base_dir_part = [args.colmap_refined_save_dir]
        base_dir_part.append(args.match_type)
        base_dir_part.append(
            args.data_part_name
        ) if args.data_part_name is not None else None
        base_dir_part.append(osp.splitext(subset_name)[0])
        base_dir_part += ["colmap_output_path", str(args.colmap_best_index)]
        args.results_save_dir = osp.join(*base_dir_part)

        logger.info("Post optimization begin!")
        state, pose_error_before_refine, pose_error_after_refine = post_optimization(
            args
        )

        if state == False:
            # Fail to construct colmap coarse data scenario
            pba.update.remote(1)
            continue

        if pose_error_before_refine is not None:
            error_before_refine[osp.splitext(subset_name)[0]] = pose_error_before_refine
        if pose_error_after_refine is not None:
            error_after_refine[osp.splitext(subset_name)[0]] = pose_error_after_refine
        pba.update.remote(1)

    return error_before_refine, error_after_refine
