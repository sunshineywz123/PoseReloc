import os
import os.path as osp
from loguru import logger

from .coarse_match.coarse_match import loftr_coarse_matching
from .coarse_sfm.coarse_sfm_runner import colmapRunner
from .post_optimization.post_optimization import post_optimization


def neuralSfM(
    img_lists, img_pairs, work_dir, enable_post_optimization=True, use_ray=False
):
    """
                                    work_dir
                                        |
        -----------------------------------------------------------------
        |           |               |             |              |
    images     keypoints.h5     matches.h5  colmap_coarse  colmap_refined
    """

    feature_out = osp.join(work_dir, "keypoints.h5")
    matche_out = osp.join(work_dir, "matches.h5") # Coarse match
    colmap_coarse_dir = osp.join(work_dir, "colmap_coarse")
    colmap_refined_dir = osp.join(work_dir, "colmap_refined")

    # LoFTR Coarse Matching:
    logger.info("LoFTR coarse matching begin...")
    loftr_coarse_matching(
        img_lists,
        img_pairs,
        feature_out=feature_out,
        match_out=matche_out,
        use_ray=use_ray,
        run_sfm_later=True,
    )

    # Coarse Mapping:
    logger.info("Coarse mapping begin...")
    colmapRunner(
        img_lists,
        img_pairs,
        work_dir,
        feature_out=feature_out,
        match_out=matche_out,
        colmap_coarse_dir=colmap_coarse_dir,
    )

    # Post Optimization
    if enable_post_optimization:
        post_optimization(
            img_lists,
            img_pairs,
            match_out_pth=matche_out,
            colmap_coarse_dir=osp.join(colmap_coarse_dir, "0"),
            refined_model_save_dir=colmap_refined_dir,
            pre_sfm=True
        )

