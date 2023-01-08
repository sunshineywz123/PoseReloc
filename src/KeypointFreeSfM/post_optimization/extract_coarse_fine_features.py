import os
import os.path as osp
import copy

from ray.actor import ActorHandle

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
from loguru import logger
import ray
from src.utils.data_io import load_obj, save_obj

from ..dataset.coarse_colmap_dataset import CoarseReconDataset
from .data_construct import MatchingPairData
from .matcher_model import *
from .feature_aggregation import feature_aggregation_and_update

cfgs = {
    "coarse_colmap_data": {
        "img_resize": 512,  # For OnePose
        "df": 8,
        "feature_track_assignment_strategy": "greedy",
        # "feature_track_assignment_strategy": "balance",
        "verbose": False,
    },
    "fine_match_debug": True,
    "fine_matcher": {
        "model": {
            "cfg_path": "configs/loftr_configs/loftr_w9_no_cat_coarse.py",
            "weight_path": "weight/loftr_w9_no_cat_coarse_auc10=0.685.ckpt",
            "seed": 666,
        },
        "visualize": False,  # Visualize fine feature map and corresponds
        # [None, 'fine_match_backbone', 'fine_match_attention'] Save for later 2D-3D match use, None means don't extract feature
        "extract_feature_method": "fine_match_backbone",
        "use_warpped_feature": False,
        "ray": {
            "slurm": False,
            "n_workers": 4,  # 4 for onepose
            "n_cpus_per_worker": 1,
            "n_gpus_per_worker": 0.25,
            "local_mode": False,
        },
    },
    "feature_aggregation_method": "avg",
    "visualize": True,  # vis3d visualize
    "evaluation": False,
}


def extract_coarse_fine_features(
    image_lists,
    covis_pairs_pth,
    colmap_coarse_dir,
    refined_model_save_dir,
    match_out_pth,
    feature_out_pth=None,  # Used to update feature
    fine_match_use_ray=False,  # Use ray for fine match
    pre_sfm=False,
    visualize_dir=None,
    vis3d_pth=None,
    verbose=True,
):
    # Overwrite some configs
    cfgs["coarse_colmap_data"]["verbose"] = verbose

    # Construct scene data
    colmap_image_dataset = CoarseReconDataset(
        cfgs["coarse_colmap_data"],
        image_lists,
        covis_pairs_pth,
        colmap_coarse_dir,
        refined_model_save_dir,
        pre_sfm=pre_sfm,
        vis_path=vis3d_pth if vis3d_pth is not None else None,
    )
    # logger.info("Scene data construct finish!")

    state = colmap_image_dataset.state
    if state == False:
        logger.warning(
            f"Build colmap coarse dataset fail! colmap point3D or images or cameras is empty!"
        )
        return state, None, None

    # Construct matching data
    matching_pairs_dataset = MatchingPairData(colmap_image_dataset)

    # Fine level match
    save_path = osp.join(match_out_pth.rsplit("/", 2)[0], "fine_matches.pkl")
    if not osp.exists(save_path) or cfgs["fine_match_debug"]:
        logger.info(f"Fine matching begin!")
        fine_match_results_dict = fine_matcher(
            cfgs["fine_matcher"],
            matching_pairs_dataset,
            use_ray=fine_match_use_ray,
            verbose=verbose,
        )
        save_obj(fine_match_results_dict, save_path)
    else:
        logger.info(f"Fine matches exists! Load from {save_path}")
        fine_match_results_dict = load_obj(save_path)

    # Update feature
    if feature_out_pth is not None:
        feature_aggregation_and_update(
            colmap_image_dataset,
            fine_match_results_dict,
            feature_out_pth=feature_out_pth,
            image_lists=image_lists,
            aggregation_method=cfgs["feature_aggregation_method"],
            verbose=verbose,
        )
    return state