from shutil import rmtree
import cv2
from loguru import logger
from vis3d.vis3d import Vis3D
from PIL import Image
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from src.utils.vis_utils import reproj as reproj_
from src.utils.vis_utils import draw_3d_box
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
        pose_pred = result_dict["pose_pred"][0]
        pose_gt = result_dict["pose_gt"]
        intrinsic = result_dict["intrinsic"]
        image_path = result_dict["image_path"]
        image0 = Image.open(image_path)
        image1 = Image.open(image_path)

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
    
        # Get draw bbox needed data:
        seq_base_path = image_path.rsplit('/color/',1)[0]
        obj_base_path = seq_base_path.rsplit('/', 1)[0]
        all_seq_name = sorted(os.listdir(obj_base_path))
        all_seq_name = [seq_name for seq_name in all_seq_name if '-' in seq_name]

        box_trans_path = osp.join(obj_base_path, all_seq_name[0], 'Box_trans.txt')
        box_path = osp.join(obj_base_path, all_seq_name[0], 'Box.txt')
        
        if osp.exists(osp.join(seq_base_path, 'intrinsics.txt')):
            intrin_full_path = osp.join(seq_base_path, 'intrinsics.txt')
        else:
            intrin_full_path = image_path.replace('/color/', '/origin_intrin/').replace('.png', '.txt')
        image_full_path = image_path.replace('/color/', '/color_full/')

        if osp.exists(image_full_path):
            if not osp.exists(box_trans_path):
                box_trans_path = None
                logger.warning(f"Box_trans.txt not exists in:{box_trans_path}")
            box3d = get_refine_box(box_path, box_trans_path)
            np.savetxt(osp.join(obj_base_path, all_seq_name[0], 'Box_transformed.txt'), box3d.T[:-1])
            K_full, K_full_homo = parse_K(intrin_full_path)
            image_full = cv2.imread(image_full_path)

            # Draw gt 3d bbox
            reproj_box_2d_gt = reproj_(K_full, pose_gt, box3d.T)
            image_full = draw_3d_box(image_full, reproj_box_2d_gt, color='g')

            # Draw pred 3d box
            if pose_pred is not None:
                reproj_box_2d_pred = reproj_(K_full, pose_pred, box3d.T)
                image_full = draw_3d_box(image_full, reproj_box_2d_pred, color='b')

            image_full_pil = Image.fromarray(cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB))
            vis3d.add_image(image_full_pil, name='results_bbox')
        else:
            box3d = get_refine_box(box_path)
            # K_full, K_full_homo = parse_K(intrin_full_path)
            image = cv2.imread(image_path)

            # Draw gt 3d bbox
            reproj_box_2d_gt = reproj_(intrinsic, pose_gt, box3d.T)
            image = draw_3d_box(image, reproj_box_2d_gt, line_width=6, color='y')

            # Draw pred 3d box
            if pose_pred is not None:
                reproj_box_2d_pred = reproj_(intrinsic, pose_pred, box3d.T)
                image = draw_3d_box(image, reproj_box_2d_pred, linewidth=6, color='g')

            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            vis3d.add_image(image_pil, name='results_bbox')