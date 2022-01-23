from vis3d.vis3d import Vis3D
from PIL import Image
import numpy as np
from ..plot_utils import reproj


def dump_obj(results_dict, vis3d_pth, vis3d_name):
    vis3d = Vis3D(vis3d_pth, vis3d_name)

    for id, result_dict in enumerate(results_dict):
        mkpts3d = result_dict["mkpts3d"]
        mkpts_query = result_dict["mkpts_query"]
        mconf = result_dict["mconf"]
        R_errs = result_dict["R_errs"]
        t_errs = result_dict["t_errs"]
        inliers = result_dict["inliers"]
        pose_pred = result_dict["pose_pred"]
        pose_gt = result_dict["pose_gt"]
        intrinsic = result_dict["intrinsic"]
        image_path = result_dict["image_path"]
        image0 = Image.open(image_path).convert('LA')
        image1 = Image.open(image_path).convert('LA')

        # Project 3D points to 2D query image
        mkpts3d_reprojed, depth = reproj(intrinsic, pose_gt, mkpts3d)

        reproj_distance = np.linalg.norm(mkpts3d_reprojed - mkpts_query, axis=-1)
        reproj_distance_minus_3 = reproj_distance < 3
        reproj_distance_minus_5 = reproj_distance < 5
        reproj_distance_minus_10 = reproj_distance < 10

        # Add vis3d
        vis3d.set_scene_id(id)
        vis3d.add_keypoint_correspondences(
            image0,
            image1,
            mkpts_query,
            mkpts3d_reprojed,
            unmatched_kpts0=None,
            unmatched_kpts1=None,
            metrics={
                "mconf": mconf.tolist(),
                "reproj_distance": reproj_distance.tolist(),
                "depth": depth.tolist()
            },
            booleans={
                "inliers": inliers[0].tolist(),
                "rpj_3pix": reproj_distance_minus_3.tolist(),
                "rpj_5pix": reproj_distance_minus_5.tolist(),
                "rpj_10pix": reproj_distance_minus_10.tolist()
            },
            meta={'R_errs': R_errs, 't_errs': t_errs},
            name="matches from reprojected 3D keypoints"
        )
