import os
import logging
import subprocess
import os.path as osp
import numpy as np

from pathlib import Path
from .colmap import read_write_model
def run_bundle_adjuster(deep_sfm_dir, ba_dir, colmap_path):
    logging.info("Running the bundle adjuster.")

    deep_sfm_model_dir = osp.join(deep_sfm_dir, 'model')
    cmd = [
        str(colmap_path), 'bundle_adjuster',
        '--input_path', str(deep_sfm_model_dir),
        '--output_path', str(ba_dir),
        '--BundleAdjustment.max_num_iterations', '150',
        '--BundleAdjustment.max_linear_solver_iterations', '500',
        '--BundleAdjustment.function_tolerance', '0',
        '--BundleAdjustment.gradient_tolerance', '0',
        '--BundleAdjustment.parameter_tolerance', '0',
        '--BundleAdjustment.refine_focal_length', '0',
        '--BundleAdjustment.refine_principal_point', '0',
        '--BundleAdjustment.refine_extra_params', '0',
        '--BundleAdjustment.refine_extrinsics', '1'
    ]
    logging.info(' '.join(cmd))

    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with point_triangulator, existing.')
        exit(ret)

def main(deep_sfm_dir, ba_dir, colmap_path='colmap'):
    assert Path(deep_sfm_dir).exists(), deep_sfm_dir

    Path(ba_dir).mkdir(parents=True, exist_ok=True)
    run_bundle_adjuster(deep_sfm_dir, ba_dir, colmap_path)

def parse_align_pose(ba_dir):
    cameras, images, points3D = read_write_model.read_model(ba_dir, ext='.bin')

    for key in images.keys():
        colmap_image = images[key]
        colmap_camera = cameras[key]

        image_name = colmap_image.name
        fx, fy, cx, cy = colmap_camera.params
        tvec, qvec = colmap_image.tvec, colmap_image.qvec

        img_type = image_name.split('/')[-2]
        img_postfix = image_name[-4:]
        ba_pose_file = image_name.replace(f'/{img_type}/', '/poses_ba/').replace(img_postfix, '.txt')
        ba_pose_dir = osp.dirname(ba_pose_file)
        Path(ba_pose_dir).mkdir(exist_ok=True, parents=True)

        # save pose
        ba_rotmat = read_write_model.qvec2rotmat(qvec)
        ba_pose = np.eye(4)
        ba_pose[:3, :3] = ba_rotmat
        ba_pose[:3, 3:] = tvec.reshape(3, 1)
        np.savetxt(ba_pose_file, ba_pose)

        # save intrin
        if img_type == 'color':
            ba_intrin_file = image_name.replace(f'/{img_type}/', '/intrin_ba/').replace(img_postfix, '.txt')
        elif img_type == 'color_full':
            ba_intrin_file = image_name.replace(f'/{img_type}/', '/intrin_full_ba/').replace(img_postfix, '.txt')

        ba_intrin_dir = osp.dirname(ba_intrin_file)
        Path(ba_intrin_dir).mkdir(exist_ok=True, parents=True)

        ba_intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        np.savetxt(ba_intrin_file, ba_intrinsic)


# def update_arkit(cfg, data_dir):
#     root_dir, sub_dirs = data_dir.split(' ')[0], data_dir.split(' ')[1:]

#     for sub_dir in sub_dirs:
#         seq_path = osp.join(root_dir, sub_dir)

#         color_full_dir = osp.join(seq_path, 'color_full')
#         intrin_full_file = osp.join(seq_path, 'intrinsics.txt')
#         intrin_ba_full_dir = osp.join(seq_path, 'intrin_full_ba')