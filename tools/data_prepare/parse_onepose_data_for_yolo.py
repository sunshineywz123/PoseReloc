import argparse
from itertools import chain
from shutil import copyfile
from tkinter import image_names
from tracemalloc import start
from typing import ChainMap
import os
import os.path as osp
import json
from git import rmtree
import numpy as np
import cv2
import math
import ray
from glob import glob
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from utils.data_utils import get_image_crop_resize, get_K_crop_resize
from utils.ray_utils import ProgressBar, chunks
from sample_points_on_cad import sample_points_on_cad
from render_cad_model_to_depth import render_cad_model_to_depth, save_np, depth2color

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input:
    parser.add_argument(
        "--data_base_dir", type=str, default="/nas/users/hexingyi/onepose_hard_data"
    )
    parser.add_argument("--obj_id", type=str, default="1")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--num_train_seq", type=int, default=1)

    # Output:
    parser.add_argument(
        "--output_data_dir", type=str, default="/nas/users/hexingyi/yolo_real_data"
    )

    args = parser.parse_args()
    return args


def save_json(json_path, data):
    with open(json_path, "w") as output_file:
        json.dump(data, output_file)


def load_json(json_pth):
    assert osp.exists(json_pth), f"json path: {json_pth} not exists!"
    with open(json_pth) as json_file:
        data = json.load(json_file)
    return data


def reproj(K, pose, pts_3d):
    """ 
    Reproj 3d points to 2d points 
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K

    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose

    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    depth = reproj_points[2]  # [N]
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T

    return reproj_points, depth  # [n, 2]


def parse_models_info_txt(models_info_txt_path):
    models_info_dict = {}
    with open(models_info_txt_path, "r") as f:
        txt_list = f.readlines()
        for obj_info in txt_list:
            obj_info_splited = obj_info.split(" ")
            obj_id = obj_info_splited.pop(0)
            model_info = {}
            for id in range(0, len(obj_info_splited), 2):
                model_info[obj_info_splited[id]] = float(obj_info_splited[id + 1])
            models_info_dict[obj_id] = model_info
    return models_info_dict


if __name__ == "__main__":
    args = parse_args()
    object_names = os.listdir(args.data_base_dir)
    id2full_name = {name[:4]: name for name in object_names if "-" in name}

    obj_full_name = id2full_name[args.obj_id]
    obj_name = obj_full_name.split('-',2)[1]

    logger.info(f"Working on obj:{obj_full_name}")

    rgb_paths = []
    if args.split == 'train':
        for seq_id in range(args.num_train_seq):
            image_seq_dir = osp.join(
                args.data_base_dir,
                obj_full_name,
                '-'.join([obj_name,str(seq_id+1)]), # FIXME: not elegant!
                'color_full'
            )
            if not osp.exists(image_seq_dir):
                logger.warning(f"{image_seq_dir} not exists!")
                continue
            rgb_paths += [osp.join(image_seq_dir, image_name) for image_name in os.listdir(image_seq_dir)]
    else:
        image_seq_dir = osp.join(
            args.data_base_dir,
            obj_full_name,
            '-'.join([obj_name,'demo']),
            'color_full'
        )
        rgb_paths += [osp.join(image_seq_dir, image_name) for image_name in os.listdir(image_seq_dir)]

    assert len(rgb_paths) != 0

    # Construct output data file structure
    output_data_base_dir = args.output_data_dir
    output_data_obj_dir = osp.join(
        output_data_base_dir,
        obj_full_name
    )
    output_image_dir = osp.join(output_data_obj_dir, "images", args.split)
    output_label_dir = osp.join(output_data_obj_dir, "labels", args.split)
    if osp.exists(output_image_dir):
        rmtree(output_image_dir)
    if osp.exists(output_label_dir):
        rmtree(output_label_dir)
    Path(output_image_dir).mkdir(parents=True, exist_ok=True,)
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)

    img_id = 0
    for global_id, rgb_path in tqdm(enumerate(rgb_paths), total=len(rgb_paths)):
        dataset_img_id, img_ext = osp.splitext(rgb_path.rsplit('/',1)[1])
        seq_path = rgb_path.rsplit('/',1)[0]
        # K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
        # pose = np.loadtxt(
        #     osp.join(image_seq_dir, "-".join([dataset_img_id, "pose"]) + ".txt")
        # )
        original_img = cv2.imread(rgb_path)
        img_h, img_w = original_img.shape[:2]

        if args.split == 'train':
            reproj_box3d_file = (
                np.loadtxt(
                    osp.join(seq_path.replace('color_full', 'reproj_box'), f"{dataset_img_id}.txt")
                )
                .astype(np.int))
            x0, y0 = reproj_box3d_file.min(0)
            x1, y1 = reproj_box3d_file.max(0)
            w, h = x1 - x0, y1 - y0

        else:
            # # In test scenario, no gt bbox available, need to render for it.
            assert NotImplementedError
            if not osp.exists(osp.join(image_seq_dir, "-".join([dataset_img_id, "box"]) + ".txt")):
                # In test scenario, no gt bbox available, need to render for it.
                depth = render_cad_model_to_depth(model_path, K, pose, original_img.shape[0], original_img.shape[1])
                obj_reg_coor_y, obj_reg_coor_x = np.where(depth!=0)
                y0, x0 = np.min(obj_reg_coor_y), np.min(obj_reg_coor_x)
                y1, x1 = np.max(obj_reg_coor_y), np.max(obj_reg_coor_x)
                w, h = x1 - x0, y1 - y0
                np.savetxt(
                    osp.join(image_seq_dir, "-".join([dataset_img_id, "box"]) + ".txt"), np.array([[x0, y0, w, h]])
                )
            else:
                x0, y0, w, h = (
                    np.loadtxt(
                        osp.join(image_seq_dir, "-".join([dataset_img_id, "box"]) + ".txt")
                    )
                    .astype(np.int)
                    .tolist()
                )
                x1, y1 = x0 + w, y0 + h

        x0_norm, y0_norm = x0 / img_w, y0 / img_h
        w_norm, h_norm = w / img_w, h / img_h
        x_center, y_center = x0_norm + w_norm * 0.5, y0_norm + h_norm * 0.5
        # Save to aim dir:
        os.symlink(rgb_path, osp.join(output_image_dir, str(global_id)+img_ext))
        # np.savetxt(osp.join(output_label_dir, img_base_name + ".txt"), [0, x_center, y_center, w_norm, h_norm])
        with open(osp.join(output_label_dir, str(global_id) + ".txt"), "w") as f:
            f.write(" ".join([str(0), str(x_center), str(y_center), str(w_norm), str(h_norm)]))
