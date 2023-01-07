import os

import os.path as osp
import numpy as np

from pathlib import Path

def parse_align_pose(ba_dir):
    from src.utils.colmap import read_write_model
    cameras, images, points3D = read_write_model.read_model(ba_dir, ext='.bin') 

    for key in images.keys():
        colmap_image = images[key]
        colmap_camera = cameras[key]

        image_name = colmap_image.name
        fx, fy, cx, cy = colmap_camera.params
        tvec, qvec = colmap_image.tvec, colmap_image.qvec

        img_type = image_name.split('/')[-2]
        ba_pose_file = image_name.replace(f'/{img_type}/', '/poses_ba/').replace('.png', '.txt')
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
            ba_intrin_file = image_name.replace(f'/{img_type}/', '/intrin_ba/').replace('.png', '.txt')
        elif img_type == 'color_full':
            ba_intrin_file = image_name.replace(f'/{img_type}/', '/intrin_full_ba/').replace('.png', '.txt')

        ba_intrin_dir = osp.dirname(ba_intrin_file)
        Path(ba_intrin_dir).mkdir(exist_ok=True, parents=True)

        ba_intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        np.savetxt(ba_intrin_file, ba_intrinsic)
