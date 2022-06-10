import argparse
from itertools import chain
from shutil import copyfile
from tkinter import image_names
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

id2name_dict = {
    1: "ape",
    2: "benchvise",
    # 3: "bowl",  # no
    4: "camera",
    5: "can",
    6: "cat",
    # 7: "cup", # no
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input:
    parser.add_argument(
        "--data_base_dir", type=str, default="/nas/users/hexingyi/lm_full"
    )
    parser.add_argument("--obj_id", type=str, default="1")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--assign_onepose_id", type=str, default="0801")
    parser.add_argument("--add_detector_noise", action='store_true')

    parser.add_argument("--use_yolo_box", action='store_true')
    parser.add_argument("--yolo_box_base_path", type=str, default="/nas/users/hexingyi/yolo_results")

    # Output:
    parser.add_argument(
        "--output_data_dir", type=str, default="/nas/users/hexingyi/onepose_hard_data"
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
    obj_name = id2name_dict[int(args.obj_id)]

    logger.info(f"Working on obj:{obj_name}")

    image_seq_dir = osp.join(
        args.data_base_dir,
        "real_train" if args.split == "train" else "real_test",
        obj_name,
    )
    model_path = osp.join(args.data_base_dir, "models", obj_name, obj_name + ".ply")
    models_info_dict = parse_models_info_txt(
        osp.join(args.data_base_dir, "models", "models_info.txt")
    )
    assert osp.exists(image_seq_dir)

    rgb_pths = glob(os.path.join(image_seq_dir, "*-color.png"))

    # Construct output data file structure
    output_data_base_dir = args.output_data_dir
    obj_full_name = "-".join([args.assign_onepose_id, "lm" + str(int(args.obj_id)), "others"])
    output_data_obj_dir = osp.join(
        output_data_base_dir,
        obj_full_name,
    )
    if (not args.add_detector_noise) and (not args.use_yolo_box):
        sequence_name = "-".join(
            ["lm" + str(int(args.obj_id)), "1" if args.split == "train" else "2"]
        )  # label seq 0 for mapping data, label seq 1 for test data
    else:
        sequence_name = "-".join(
            ["lm" + str(int(args.obj_id)), "4"]
        )  # label seq 0 for mapping data, label seq 1 for test data
    output_data_seq_dir = osp.join(output_data_obj_dir, sequence_name,)
    if osp.exists(output_data_seq_dir):
        rmtree(output_data_seq_dir)
    Path(output_data_seq_dir).mkdir(parents=True, exist_ok=True)

    color_path = osp.join(output_data_seq_dir, "color")
    color_full_path = osp.join(output_data_seq_dir, "color_full")
    intrin_path = osp.join(output_data_seq_dir, "intrin_ba")
    intrin_origin_path = osp.join(output_data_seq_dir, "intrin")
    poses_path = osp.join(output_data_seq_dir, "poses_ba")
    Path(color_path).mkdir(exist_ok=True)
    Path(color_full_path).mkdir(exist_ok=True)
    Path(intrin_path).mkdir(exist_ok=True)
    Path(intrin_origin_path).mkdir(exist_ok=True)
    Path(poses_path).mkdir(exist_ok=True)

    # Save model info:
    if args.split == "train":
        model_min_xyz = np.array(
            [
                models_info_dict[str(int(args.obj_id))]["min_x"],
                models_info_dict[str(int(args.obj_id))]["min_y"],
                models_info_dict[str(int(args.obj_id))]["min_z"],
            ]
        )
        model_size_xyz = np.array(
            [
                models_info_dict[str(int(args.obj_id))]["size_x"],
                models_info_dict[str(int(args.obj_id))]["size_y"],
                models_info_dict[str(int(args.obj_id))]["size_z"],
            ]
        )
        model_max_xyz = model_min_xyz + model_size_xyz
        # if np.linalg.norm(model_min_xyz) > np.linalg.norm(model_max_xyz):
        #     extend_xyz = np.abs(model_min_xyz)
        # else:
        #     extend_xyz = np.abs(model_max_xyz)
        extend_xyz = (model_max_xyz - model_min_xyz) / 1000 # convert to m

        extend_xyz_str = ",".join(
            np.concatenate([np.array([0, 0, 0]), extend_xyz, np.array([0, 0, 0, 0])])
            .astype(str)
            .tolist()
        )
        with open(osp.join(output_data_seq_dir, "Box.txt"), "w") as f:
            f.write(
                "# px(position_x), py, pz, ex(extent_x), ey, ez, qw(quaternion_w), qx, qy, qz\n"
            )
            f.write(extend_xyz_str)

        # Copy eval model and save diameter:
        model_eval_path = model_path
        diameter = models_info_dict[str(int(args.obj_id))]["diameter"] / 1000  # convert to m
        assert osp.exists(
            model_eval_path
        ), f"model eval path:{model_eval_path} not exists!"
        copyfile(model_eval_path, osp.join(output_data_obj_dir, "model_eval.ply"))
        np.savetxt(osp.join(output_data_obj_dir, "diameter.txt"), np.array([diameter]))


    img_id = 0
    for global_id, image_path in tqdm(enumerate(rgb_pths), total=len(rgb_pths)):
        dataset_img_id, file_label = (
            osp.splitext(image_path)[0].rsplit("/", 1)[1].split("-")
        )
        img_ext = osp.splitext(image_path)[1]
        K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
        pose = np.loadtxt(
            osp.join(image_seq_dir, "-".join([dataset_img_id, "pose"]) + ".txt")
        )
        original_img = cv2.imread(image_path)
        img_h,img_w = original_img.shape[:2]

        if args.split == 'train':
            # Load GT box directly:
            x0, y0, w, h = (
                np.loadtxt(
                    osp.join(image_seq_dir, "-".join([dataset_img_id, "box"]) + ".txt")
                )
                .astype(np.int)
                .tolist()
            )
            x1, y1 = x0 + w, y0 + h

        else:
            if args.use_yolo_box:
                yolo_box_base_path = args.yolo_box_base_path
                yolo_box_obj_path = osp.join(yolo_box_base_path, args.split, obj_full_name, 'labels')
                yolo_box = np.loadtxt(osp.join(yolo_box_obj_path, dataset_img_id+'.txt'))
                assert yolo_box.shape[0] != 0, f"img id:{dataset_img_id} no box detected!"
                if len(yolo_box.shape) == 2:
                    # not only box! select by maxium confidence
                    want_id = np.argsort(yolo_box[:,5])[0]
                    yolo_box = yolo_box[want_id]
                
                x_c_n, y_c_n, w_n, h_n = yolo_box[1:5]
                x0_n, y0_n = x_c_n - w_n / 2, y_c_n - h_n /2

                x0, y0, w, h = int(x0_n * img_w), int(y0_n * img_h), int(w_n * img_w), int(h_n * img_h)
                x1, y1 = x0 + w, y0 + h

                # x0_gt, y0_gt, w_gt, h_gt = (
                #     np.loadtxt(
                #         osp.join(image_seq_dir, "-".join([dataset_img_id, "box"]) + ".txt")
                #     )
                #     .astype(np.int)
                #     .tolist()
                # )

            else:
                # Use GT box
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


        if not args.add_detector_noise:
            compact_percent = 0.3
            # offset_percent = 0.5
            x0 -= int(w * compact_percent)
            y0 -= int(h * compact_percent)
            x1 += int(w * compact_percent)
            y1 += int(h * compact_percent)
        else:
            # compact_percent = np.random.uniform(low=0.1, high=0.4)
            compact_percent = 0.3
            offset_percent = np.random.uniform(low=-1*compact_percent, high=1*compact_percent)
            # apply compact noise:
            x0 -= int(w * compact_percent)
            y0 -= int(h * compact_percent)
            x1 += int(w * compact_percent)
            y1 += int(h * compact_percent)
            # apply offset noise:
            x0 += int(w * offset_percent)
            y0 += int(h * offset_percent)
            x1 += int(w * offset_percent)
            y1 += int(h * offset_percent)

        # Crop image by 2D visible bbox, and change K
        box = np.array([x0, y0, x1, y1])
        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop, K_crop_homo = get_K_crop_resize(box, K, resize_shape)
        image_crop, _ = get_image_crop_resize(original_img, box, resize_shape)

        box_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([256, 256])  # FIXME: change to global configs
        # resize_shape = np.array([512, 512])  # FIXME: change to global configs
        K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
        image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)

        # Save to aim dir:
        cv2.imwrite(osp.join(color_path, str(global_id) + img_ext), image_crop)
        cv2.imwrite(osp.join(color_full_path, str(global_id) + img_ext), original_img)
        np.savetxt(osp.join(intrin_path, str(global_id) + ".txt"), K_crop)
        np.savetxt(osp.join(intrin_origin_path, str(global_id) + ".txt"), K) # intrin full
        np.savetxt(osp.join(poses_path, str(global_id) + ".txt"), pose)
