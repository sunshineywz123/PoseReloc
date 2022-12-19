import argparse
from itertools import chain
from typing import ChainMap
import os
import os.path as osp
import json
import numpy as np
import cv2
import math
import ray
from tqdm import tqdm

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

    # Output:
    parser.add_argument(
        "--output_json_dir", type=str, default="/nas/users/hexingyi/bop_json"
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
                    img_path = osp.join(img_dir, "0" * (6 - len(img_id)) + img_id + ".jpg")
                    assert osp.exists(img_path), f"{img_path} not exists!"
                except:
                    img_path = osp.join(img_dir, "0" * (6 - len(img_id)) + img_id + ".png")
                    assert osp.exists(img_path), f"{img_path} not exists!"

                if visib_fract < 0.4:
                    # No enough visible part used for train
                    break

                # Load CAD model and sample points
                cat_model_path = osp.join(
                    model_dir, "obj_" + "0" * (6 - len(obj_id)) + obj_id + ".ply",
                )
                assert osp.exists(cat_model_path)

                points_sampled, _ = sample_points_on_cad(cat_model_path, 5000)
                sampled_points_rpj, sampled_depth = reproj(
                    K, np.concatenate([R_m2c, t_m2c], axis=-1), points_sampled
                )

                result_dic[
                    "###".join(
                        [
                            cat_model_path.replace(data_base_dir + "/", ""),
                            img_path.replace(data_base_dir + "/", ""),
                        ]
                    )
                ] = {
                    "obj_index": obj_index,  # obj_index of image obj list
                    "R_m2c": R_m2c.tolist(),
                    "t_m2c": t_m2c.tolist(),
                    "K": K.tolist(),
                    "bbox_obj": bbox_obj,
                    "bbox_visible": bbox_visible,
                    "split_patten": "###",  # Used for split cat model path and image path from dict key.
                }

                if save_obj_depth:
                    # Render depth according to object:
                    image = cv2.imread(img_path)
                    H, W = image.shape[:2]
                    depth_range = np.max(sampled_depth) - np.min(sampled_depth)

                    depth = render_cad_model_to_depth(
                        cat_model_path,
                        K,
                        [R_m2c, t_m2c],
                        H,
                        W,
                        # depth_img_save_path="depth.png",
                        depth_range_prior=[
                            np.min(sampled_depth) - 0.1 * depth_range,
                            np.max(sampled_depth) + 0.1 * depth_range,
                        ],
                    )

                    # Load semantic mask, mask depth and save
                    mask_visib_path = osp.join(
                        mask_visib_dir,
                        "_".join(
                            [
                                "0" * (6 - len(img_id)) + img_id,
                                "0" * (6 - len(obj_index)) + obj_index,
                            ]
                        )
                        + ".png",
                    )
                    mask_visib = cv2.imread(
                        mask_visib_path, flags=cv2.IMREAD_GRAYSCALE
                    ).astype(np.bool)
                    depth[~mask_visib] = 0

                    obj_depth_path = osp.join(
                        obj_depth_dir,
                        "_".join(
                            [
                                "0" * (6 - len(img_id)) + img_id,
                                "0" * (6 - len(obj_index)) + obj_index,
                            ]
                        )
                        + ".npy",
                    )
                    save_np(depth, obj_depth_path)

                    result_dic[
                        "###".join(
                            [
                                cat_model_path.replace(data_base_dir + "/", ""),
                                img_path.replace(data_base_dir + "/", ""),
                            ]
                        )
                    ].update(
                        {
                            "obj_depth_relative_path": osp.join(
                                obj_depth_dir,
                                "_".join(
                                    [
                                        "0" * (6 - len(img_id)) + img_id,
                                        "0" * (6 - len(obj_index)) + obj_index,
                                    ]
                                )
                                + ".npy",
                            ).replace(data_base_dir + "/", ""),
                        }
                    )
                    # Debug visual color depth:
                    # color_depth = depth2color(depth)
                    # color_depth.save('depth.png')

                # Only exists once, don't need further search for current frame
                break

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
    image_seq_dir = osp.join(args.data_base_dir, args.dataset_name, "train_pbr" if args.split == "train" else "test")

    seq_dir_list = os.listdir(image_seq_dir)
    seq_dir_list = [
        seq_name for seq_name in seq_dir_list if ".DS_Store" not in seq_name
    ]

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

    output_json_dir = args.output_json_dir
    os.makedirs(output_json_dir, exist_ok=True)
    save_json(
        osp.join(
            output_json_dir, f"{args.split}_{args.dataset_name}_{args.obj_id}.json"
        ),
        results_dict,
    )  # datasetName_objID.json
