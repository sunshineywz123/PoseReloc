import os
import os.path as osp
import copy

from ray.actor import ActorHandle

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
from loguru import logger
import ray
from src.utils.data_utils import load_obj, save_obj

from src.datasets.neuralsfm_coarse_colmap_dataset import CoarseColmapDataset
from .data_construct import (
    MatchingPairData,
    ConstructOptimizationData,
)
from .matcher_model import *
from .optimizer.optimizer import Optimizer


def post_optimization(args):
    # Construct scene data
    colmap_image_dataset = CoarseColmapDataset(
        args.data_root,
        args.colmap_results_load_dir,
        args.raw_match_results_dir,
        args.results_save_dir,
        args.img_resize_max,
        args.n_imgs,
        subset_name=args.subset_name,
    )
    logger.info("Scene data construct finish!")

    state = colmap_image_dataset.state
    if state == False:
        logger.warning(
            f"Build colmap coarse dataset fail! colmap point3D or images or cameras is empty!"
        )
        return state, None, None

    # Construct matching data
    matching_pairs = MatchingPairData(colmap_image_dataset)

    # Fine level match
    save_path = osp.join(args.scene_results_base_dir, "raw_matches/fine_matches.pkl")
    if not osp.exists(save_path) or args.fine_match_debug:
        logger.info(f"Fine matching begin!")
        detector, matcher = build_model(args)
        subset_ids = range(len(matching_pairs))
        fine_match_results = matchWorker(
            matching_pairs, subset_ids, detector, matcher, debug=False, args={},
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
    # TODO: move to global configs
    optimizer = Optimizer(optimization_data)
    optimization_procedure = ["depth", "pose"] * 3
    optimization_procedure += ["BA"]
    # optimization_procedure = ['BA']
    results_dict = optimizer(optimization_procedure)

    # Update results
    (
        pose_error_before_refine,
        pose_error_after_refine,
    ) = colmap_image_dataset.update_optimize_results_to_colmap(
        results_dict, visualize=args.visualize, evaluation=args.evaluation
    )

    return state, pose_error_before_refine, pose_error_after_refine


@ray.remote(num_cpus=1, num_gpus=1, max_calls=1)  # release gpu after finishingI
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
        state, pose_error_before_refine, pose_error_after_refine = post_optimization(args)

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
