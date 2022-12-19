import argparse
from itertools import chain
from shutil import copyfile, rmtree
from typing import ChainMap
import os
import os.path as osp
import json
from more_itertools import sample
import numpy as np
from loguru import logger
import cv2
import math
import ray
from tqdm import tqdm
from sample_points_on_cad import sample_points_on_cad, model_diameter_from_bbox

dataset_name2model_dict = {
    "ycbv": "models",
    "lm": "models",
    "tless": "models_cad",
}
ext_bag = [".png", ".jpg"]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input:
    parser.add_argument(
        "--data_base_dir", type=str, default="/nas/users/hexingyi/onepose_hard_data"
    )
    parser.add_argument("--obj_id", type=str, default="0600")
    parser.add_argument("--train_seq_ids", type=str, default="1", help="split by ,")
    parser.add_argument("--val_seq_ids", type=str, default="2", help="split by ,")

    # Output:
    parser.add_argument(
        "--output_dir", type=str, default="/nas/users/hexingyi/pvnet_data"
    )

    # Ray related
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--n_workers", type=int, default=15)
    parser.add_argument("--n_cpus_per_worker", type=float, default=2)
    parser.add_argument(
        "--local_mode", action="store_true", help="ray local mode for debugging."
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


if __name__ == "__main__":
    # NOTE: need to run parse onepose data for obj first to generate mask for each image
    args = parse_args()
    data_base_dir = args.data_base_dir
    all_obj_names = os.listdir(data_base_dir)
    id2full_name = {name[:4]: name for name in all_obj_names if "-" in name}

    # Get obj full name(obj dir name) from obj id
    assert args.obj_id in id2full_name
    id_name_cato = id2full_name[args.obj_id]
    id, obj_name, cato = id_name_cato.split("-", 2)

    # Merge images from multiple seqs:
    train_seq_ids = args.train_seq_ids.split(",")  # List [id0, id1...]
    train_seq_names = ["-".join([obj_name, seq_id]) for seq_id in train_seq_ids]
    val_seq_ids = args.val_seq_ids.split(",")  # List [id0, id1...]
    val_seq_names = ["-".join([obj_name, seq_id]) for seq_id in val_seq_ids]

    seq_names = train_seq_names + val_seq_names
    tags = ["train"] * len(train_seq_names) + ["val"] * len(val_seq_names)

    # Combine multiple sequence of one object to one seq:
    img_lists = []
    split_lists = []
    ext_bag = [".png", ".jpg"]
    for id, seq_name in enumerate(seq_names):
        seq_dir = osp.join(data_base_dir, id_name_cato, seq_name)
        img_name_lists = os.listdir(osp.join(seq_dir, "color"))
        seq_img_lists = [
            osp.join(seq_dir, "color", img_name)
            for img_name in img_name_lists
            if osp.splitext(img_name)[1] in ext_bag
        ]
        img_lists += seq_img_lists
        split_lists += [tags[id]] * len(seq_img_lists)

    # Make destination dirs:
    output_dir = osp.join(args.output_dir, "_".join([args.obj_id, "train", "val"]))
    dst_model_path = osp.join(output_dir, "model.ply")
    dst_diameter_path = osp.join(output_dir, "diameter.txt")
    dst_image_dir = osp.join(output_dir, "rgb")
    dst_pose_dir = osp.join(output_dir, "pose")
    dst_intrin_dir = osp.join(output_dir, "intrin")
    dst_mask_dir = osp.join(output_dir, "mask")
    if osp.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dst_image_dir, exist_ok=True)
    os.makedirs(dst_pose_dir, exist_ok=True)
    os.makedirs(dst_intrin_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    # Copy!
    copyfile(
        osp.join(data_base_dir, id_name_cato, "model.ply"), dst_model_path
    )  # Copy model
    _, model_bbox_corner = sample_points_on_cad(osp.join(data_base_dir, id_name_cato, "model.ply"))
    diameter = model_diameter_from_bbox(model_bbox_corner)
    print(f"Model diameter is:{diameter}")
    np.savetxt(dst_diameter_path, np.array([diameter]))
    for global_id, img_path in tqdm(enumerate(img_lists), total=len(img_lists)):
        img_ext = osp.splitext(img_path)[1]
        copyfile(
            img_path,
            osp.join(
                dst_image_dir,
                "_".join([split_lists[global_id], str(global_id)]) + img_ext,
            ),
        )
        copyfile(
            img_path.replace("/color/", "/poses_ba/").replace(img_ext, ".txt"),
            osp.join(dst_pose_dir, f"pose{str(global_id)}.txt"),
        )
        copyfile(
            img_path.replace("/color/", "/intrin_ba/").replace(img_ext, ".txt"),
            osp.join(dst_intrin_dir, str(global_id) + ".txt"),
        )
        copyfile(
            img_path.replace("/color/", "/obj_mask/").replace(img_ext, ".png"),
            osp.join(dst_mask_dir, str(global_id) + ".png"),
        )

