from shutil import rmtree
from unittest import result
import cv2
from loguru import logger
from vis3d.vis3d import Vis3D
from PIL import Image
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from src.utils.objScanner_utils import get_refine_box, parse_K
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

def dump_obj_with_feature_map(results_dict, vis3d_pth, vis3d_name, fast=False):
    vis3d = Vis3D(vis3d_pth, vis3d_name)
    if osp.exists(osp.join(vis3d_pth,vis3d_name)):
        rmtree(osp.join(vis3d_pth,vis3d_name))

    for id, result_dict in tqdm(enumerate(results_dict)):
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

        # feature 3D 2D part:
        point_cloud = result_dict['point_cloud']
        feature3D_color_before = result_dict["feature3D_color_before"]
        feature2D_color_before = result_dict["feature2D_color_before"]
        feature3D_color_after = result_dict["feature3D_color_after"]
        feature2D_color_after = result_dict["feature2D_color_after"]

        # Project 3D points to 2D query image
        mkpts3d_reprojed, depth = reproj(intrinsic, pose_gt, mkpts3d)

        reproj_distance = np.linalg.norm(mkpts3d_reprojed - mkpts_query, axis=-1)
        reproj_distance_minus_3 = reproj_distance < 3
        reproj_distance_minus_5 = reproj_distance < 5
        reproj_distance_minus_10 = reproj_distance < 10

        # Add vis3d
        vis3d.set_scene_id(id)
        if not fast:
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

        # Add: pointcloud:
        # import ipdb; ipdb.set_trace()

        if not fast:
            vis3d.add_point_cloud(vertices=point_cloud, colors=feature3D_color_before, name='point3D_feature_before')
            vis3d.add_point_cloud(vertices=point_cloud, colors=feature3D_color_after, name='point3D_feature_after')
            vis3d.add_image(feature2D_color_before, name='feature2D_before')
            vis3d.add_image(feature2D_color_after, name='feature2D_after')


        if osp.exists(image_full_path):
            if not osp.exists(box_trans_path):
                box_trans_path = None
                logger.warning(f"Box_trans.txt not exists in:{box_trans_path}")

            box3d = get_refine_box(box_path, box_trans_path)
            np.savetxt(osp.join(obj_base_path, all_seq_name[0], 'Box_transformed.txt'), box3d.T[:-1])
            img_ext = osp.splitext(image_path)[1]
            # For LINEMOD:
            # K_full = np.loadtxt(image_path.replace('/color/', '/intrin/').replace(img_ext, '.txt'))
            # For onepose data
            K_full, K_full_homo = parse_K(intrin_full_path)
            image_full = cv2.imread(image_full_path)

            # Draw gt 3d bbox
            reproj_box_2d_gt = reproj_(K_full, pose_gt, box3d.T)
            image_full = draw_3d_box(image_full, reproj_box_2d_gt, linewidth=15, color='g')

            # Draw pred 3d box
            # if pose_pred is not None:
            #     reproj_box_2d_pred = reproj_(K_full, pose_pred, box3d.T)
            #     image_full = draw_3d_box(image_full, reproj_box_2d_pred, linewidth=15, color='b')

            image_full_pil = Image.fromarray(cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB))
            vis3d.add_image(image_full_pil, name='results_bbox')

            result_image_dir = osp.join(vis3d_pth, vis3d_name, 'bbox_images')
            result_pose_dir = osp.join(vis3d_pth, vis3d_name, 'pred_poses')
            os.makedirs(result_image_dir, exist_ok=True)
            os.makedirs(result_pose_dir, exist_ok=True)
            image_name = osp.basename(image_full_path)
            pose_file_name = osp.splitext(image_name)[0] + '.txt'
            image_full_pil.save(osp.join(result_image_dir, image_name))
            np.savetxt(osp.join(result_pose_dir, pose_file_name), pose_pred)
        else:
            box3d = get_refine_box(box_path)
            # K_full, K_full_homo = parse_K(intrin_full_path)
            image = cv2.imread(image_path)

            # Draw gt 3d bbox
            reproj_box_2d_gt = reproj_(intrinsic, pose_gt, box3d.T)
            image = draw_3d_box(image, reproj_box_2d_gt, color='y')

            # Draw pred 3d box
            if pose_pred is not None:
                reproj_box_2d_pred = reproj_(intrinsic, pose_pred, box3d.T)
                image = draw_3d_box(image, reproj_box_2d_pred, color='g')

            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            vis3d.add_image(image_pil, name='results_bbox')