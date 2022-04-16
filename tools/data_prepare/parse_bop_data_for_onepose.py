import argparse
from itertools import chain
from shutil import copyfile
from typing import ChainMap
import os
import os.path as osp
import json
import numpy as np
import cv2
import math
import ray
from pathlib import Path
from tqdm import tqdm

from utils.data_utils import get_image_crop_resize, get_K_crop_resize
from utils.ray_utils import ProgressBar, chunks
from sample_points_on_cad import sample_points_on_cad
from render_cad_model_to_depth import render_cad_model_to_depth, save_np, depth2color

dataset_name2model_dict = {
    "ycbv": "models",
    "lm": "models",
    "tless": "models_cad",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input:
    parser.add_argument("--data_base_dir", type=str, default="/nas/users/hexingyi/bop")
    parser.add_argument("--dataset_name", type=str, default="ycbv")
    parser.add_argument("--obj_id", type=str, default="1")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--assign_onepose_id", type=str, default="0700")

    # Output:
    parser.add_argument(
        "--output_data_dir", type=str, default="/nas/users/hexingyi/onepose_hard_data"
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


def parse_data_for_obj(
    data_base_dir, dataset_name, seq_name, obj_id, split, verbose=False
):
    """
    data_base_dir:
        - dataset_name0
            - model
            - train_pbr
                - seq_name0
                    - rgb
                    - mask_visib
                    - obj_depth (generated by this func by setting save_obj_depth flag to `True`)
                - seq_name1
    """
    save_obj_depth = (
        split == "train"
    )  # save obj depth for making corrspondence GT in training
    seq_dir = osp.join(
        data_base_dir,
        dataset_name,
        "train_pbr" if split == "train" else "test",
        seq_name,
    )
    gt_pose_dic = load_json(osp.join(seq_dir, "scene_gt.json"))
    gt_meta_dic = load_json(osp.join(seq_dir, "scene_gt_info.json"))
    gt_camera_dic = load_json(osp.join(seq_dir, "scene_camera.json"))

    img_dir = osp.join(seq_dir, "rgb")
    mask_visib_dir = osp.join(seq_dir, "mask_visib")
    assert dataset_name in dataset_name2model_dict
    model_dir_name = dataset_name2model_dict[dataset_name]
    model_dir = osp.join(data_base_dir, dataset_name, model_dir_name)
    model_eval_dir = osp.join(data_base_dir, dataset_name, "models_eval")
    models_info = load_json(osp.join(model_dir, "models_info.json"))
    model_info = models_info[obj_id]

    min_xyz = [model_info["min_x"], model_info["min_y"], model_info["min_z"]]
    size_xyz = [model_info["size_x"], model_info["size_y"], model_info["size_z"]]
    diameter = model_info["diameter"]

    obj_depth_dir = osp.join(seq_dir, "obj_depth")
    os.makedirs(obj_depth_dir, exist_ok=True)
    assert osp.exists(img_dir) and osp.exists(mask_visib_dir) and osp.exists(model_dir)

    result_dic = {}
    gt_pose_dic = tqdm(gt_pose_dic.items()) if verbose else gt_pose_dic.items()
    for global_id, (img_id, pose_info) in enumerate(gt_pose_dic):
        # if global_id > 100:
        #     break
        for i, obj_pose in enumerate(pose_info):
            if obj_id == str(obj_pose["obj_id"]):
                # Find target obj in current image
                obj_index = str(
                    i
                )  # index of the obj in image, useful for parse semantic mask
                R_m2c = np.array(obj_pose["cam_R_m2c"]).reshape(3, 3)  # 3*3
                t_m2c = np.array(obj_pose["cam_t_m2c"])[:, None]  # 3*1
                K = np.array(gt_camera_dic[img_id]["cam_K"]).reshape(3, 3)  # 3*3
                bbox_obj = gt_meta_dic[img_id][i]["bbox_obj"]
                bbox_visible = gt_meta_dic[img_id][i]["bbox_visib"]
                visib_fract = gt_meta_dic[img_id][i]["visib_fract"]
                try:
                    img_path = osp.join(
                        img_dir, "0" * (6 - len(img_id)) + img_id + ".jpg"
                    )
                    assert osp.exists(img_path), f"{img_path} not exists!"
                except:
                    img_path = osp.join(
                        img_dir, "0" * (6 - len(img_id)) + img_id + ".png"
                    )
                    assert osp.exists(img_path), f"{img_path} not exists!"

                if visib_fract < 0.4:
                    # No enough visible part used for train
                    break

                # Load CAD model and sample points
                cad_model_path = osp.join(
                    model_dir, "obj_" + "0" * (6 - len(obj_id)) + obj_id + ".ply",
                )
                cad_model_eval_path = osp.join(
                    model_eval_dir, "obj_" + "0" * (6 - len(obj_id)) + obj_id + ".ply",
                )

                assert osp.exists(cad_model_path)

                # points_sampled, _ = sample_points_on_cad(cat_model_path, 5000)
                # sampled_points_rpj, sampled_depth = reproj(
                #     K, np.concatenate([R_m2c, t_m2c], axis=-1), points_sampled
                # )

                assert img_path not in result_dic
                result_dic[img_path] = {
                    "obj_index": obj_index,  # obj_index of image obj list
                    "R_m2c": R_m2c.tolist(),
                    "t_m2c": t_m2c.tolist(),
                    "K": K.tolist(),
                    "bbox_obj": bbox_obj,
                    "bbox_visible": bbox_visible,
                    "cad_model_path": cad_model_path,
                    "cad_model_eval_path": cad_model_eval_path,
                    "model_diameter": diameter,
                    "model_min_xyz": min_xyz,
                    "model_size_xyz": size_xyz,
                }
                break

                # TODO: get mask path

    return result_dic


def parse_data_for_obj_multiple_seqs(
    data_base_dir, dataset_name, seq_dir_list, obj_id, split, pba=None
):
    results_seq_list = []
    seq_dir_list = tqdm(seq_dir_list) if pba is None else seq_dir_list
    for seq_name in seq_dir_list:
        result_dir = parse_data_for_obj(
            data_base_dir, dataset_name, seq_name, obj_id=obj_id, split=split,
        )
        results_seq_list += [result_dir]

        if pba is not None:
            pba.update.remote(1)
    return results_seq_list


@ray.remote(num_cpus=2)
def parse_data_for_obj_multiple_seqs_ray_wrapper(*args, **kwargs):
    return parse_data_for_obj_multiple_seqs(*args, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    image_seq_dir = osp.join(
        args.data_base_dir,
        args.dataset_name,
        "train_pbr" if args.split == "train" else "test",
    )

    seq_dir_list = os.listdir(image_seq_dir)
    seq_dir_list = [
        seq_name for seq_name in seq_dir_list if ".DS_Store" not in seq_name
    ]

    # Construct output data file structure
    output_data_base_dir = args.output_data_dir
    output_data_obj_dir = osp.join(
        output_data_base_dir,
        "-".join([args.assign_onepose_id, args.dataset_name + args.obj_id, "others"]),
    )
    sequence_name = "-".join(
        [args.dataset_name + args.obj_id, "1" if args.split == "train" else "2"]
    )  # label seq 0 for mapping data, label seq 1 for test data
    output_data_seq_dir = osp.join(output_data_obj_dir, sequence_name,)
    Path(output_data_seq_dir).mkdir(parents=True, exist_ok=True)

    if args.use_ray:
        if args.slurm:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(args.n_workers * args.n_cpus_per_worker),
                # num_gpus=math.ceil(args.n_workers * args.n_gpus_per_worker),
                local_mode=args.local_mode,
            )

        pb = ProgressBar(
            len(seq_dir_list),
            f"Parse bop data set: {args.dataset_name}, obj:{args.obj_id}...",
        )
        all_subsets = chunks(
            seq_dir_list, math.ceil(len(seq_dir_list) / args.n_workers)
        )

        results = [
            parse_data_for_obj_multiple_seqs_ray_wrapper.remote(
                args.data_base_dir,
                args.dataset_name,
                subsets,
                obj_id=args.obj_id,
                split=args.split,
                pba=pb.actor,
            )
            for subsets in all_subsets
        ]
        pb.print_until_done()
        results_seq_list = list(chain(*ray.get(results)))
    else:
        results_seq_list = parse_data_for_obj_multiple_seqs(
            args.data_base_dir,
            args.dataset_name,
            seq_dir_list,
            obj_id=args.obj_id,
            split=args.split,
            pba=None,
        )

    results_dict = dict(ChainMap(*results_seq_list))
    color_path = osp.join(output_data_seq_dir, "color")
    intrin_path = osp.join(output_data_seq_dir, "intrin_ba")
    poses_path = osp.join(output_data_seq_dir, "poses_ba")
    Path(color_path).mkdir(exist_ok=True)
    Path(intrin_path).mkdir(exist_ok=True)
    Path(poses_path).mkdir(exist_ok=True)

    for global_id, (image_path, image_info) in tqdm(
        enumerate(results_dict.items()), total=len(results_dict)
    ):

        if global_id == 0 and args.split == "train":
            # Make bbox info
            model_min_xyz = np.array(image_info["model_min_xyz"])
            model_size_xyz = np.array(image_info["model_size_xyz"])
            model_max_xyz = model_min_xyz + model_size_xyz
            # if np.linalg.norm(model_min_xyz) > np.linalg.norm(model_max_xyz):
            #     extend_xyz = np.abs(model_min_xyz)
            # else:
            #     extend_xyz = np.abs(model_max_xyz)
            extend_xyz = model_max_xyz - model_min_xyz

            extend_xyz_str = ",".join(
                np.concatenate(
                    [np.array([0, 0, 0]), extend_xyz, np.array([0, 0, 0, 0])]
                )
                .astype(str)
                .tolist()
            )
            with open(osp.join(output_data_seq_dir, "Box.txt"), "w") as f:
                f.write(
                    "# px(position_x), py, pz, ex(extent_x), ey, ez, qw(quaternion_w), qx, qy, qz\n"
                )
                f.write(extend_xyz_str)

            # Copy eval model and save diameter:
            model_eval_path = image_info["cad_model_eval_path"]
            diameter = image_info['model_diameter']
            assert osp.exists(
                model_eval_path
            ), f"model eval path:{model_eval_path} not exists!"
            copyfile(model_eval_path, osp.join(output_data_obj_dir, 'model_eval.ply'))
            np.savetxt(osp.join(output_data_obj_dir, 'diameter.txt'), np.array([diameter]))

        img_ext = osp.splitext(image_path)[1]
        K = np.array(image_info["K"])  # 3*3
        R = np.array(image_info["R_m2c"])  # 3*3
        t = np.array(image_info["t_m2c"])  # 3*1
        pose = np.concatenate(
            [np.concatenate([R, t], axis=1), np.array([[0, 0, 0, 1]])], axis=0
        )  # 4*4
        original_img = cv2.imread(image_path)
        x0, y0, w, h = image_info["bbox_visible"]
        x1, y1 = x0 + w, y0 + h

        # Crop image by 2D visible bbox, and change K
        box = np.array([x0, y0, x1, y1])
        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop, K_crop_homo = get_K_crop_resize(box, K, resize_shape)
        image_crop, _ = get_image_crop_resize(original_img, box, resize_shape)

        box_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([512, 512])  # FIXME: change to global configs
        K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
        image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)

        # TODO: add color full, intrin full

        # Save to aim dir:
        cv2.imwrite(osp.join(color_path, str(global_id) + img_ext), image_crop)
        np.savetxt(osp.join(intrin_path, str(global_id) + ".txt"), K_crop)
        np.savetxt(osp.join(poses_path, str(global_id) + ".txt"), pose)
