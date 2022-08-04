import argparse
from asyncio.log import logger
import os
import os.path as osp
import open3d as o3d
from tqdm import tqdm
import numpy as np
from tools.data_prepare.sample_points_on_cad import (
    model_diameter_from_bbox,
    get_model_corners,
    sample_points_on_cad,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input:
    parser.add_argument(
        "--data_base_path", type=str, default="/nas/users/hexingyi/onepose_hard_data",
    )

    parser.add_argument(
        "--sfm_base_path",
        type=str,
        default="/nas/users/hexingyi/transfer/onepose_sfm_v3/outputs_softmax_loftr_loftr",
    )
    parser.add_argument(
        "--obj_ids", nargs="+", default=["0600", '0601', '0604', '0606', '0607', '0623', '0627', '0639', '0640', '0641'],
        # "--obj_ids", nargs="+", default=["0801", "0802", "0804", "0805","0806", "0808", "0809" ],
    )
    parser.add_argument(
        "--threshold", nargs="+", type=int, default=[1, 3, 5], help="threshold unit: mm"
    )
    parser.add_argument("--model_unit", type=str, default="m", choices=["m", "mm"])

    args = parser.parse_args()
    return args


def compute_pointcloud_accu_and_complete(
    gt_mesh_path, pred_pointcloud_path, thrs=[1, 3, 5], model_unit="m", max_gt_num=50000
):
    """
    thrs unit: mm
    """
    assert osp.exists(gt_mesh_path), f"{gt_mesh_path}"
    assert osp.exists(pred_pointcloud_path), f"{pred_pointcloud_path}"
    # Load GT mesh:
    # mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    # gt_pointcloud = np.asarray(mesh.vertices)  # N*3
    gt_pointcloud, _ = sample_points_on_cad(gt_mesh_path, max_gt_num)
    pred_pointcloud = o3d.io.read_point_cloud(pred_pointcloud_path)
    pred_pointcloud = np.asarray(pred_pointcloud.points)  # M*3
    print(f'Total: {pred_pointcloud.shape[0]} 3D points')

    # Compute distance:
    completeness_multiple_thr = {}
    accuracy_multiple_thr = {}
    distance_matrix = np.linalg.norm(
        gt_pointcloud[:, None] - pred_pointcloud[None], axis=-1
    )  # N*M
    for distance_thr in thrs:
        if model_unit == "m":
            thr_scale = 1000  # convert thr's unit from mm to m
        elif model_unit == "cm":
            thr_scale = 10
        elif model_unit == "mm":
            thr_scale = 1
        else:
            raise NotImplementedError
        distance_mask = distance_matrix < (distance_thr / thr_scale)
        completeness_count = np.sum(distance_mask, axis=1)  # horizontal count, N*3
        accuracy_count = np.sum(distance_mask, axis=0)  # vertical count, M*3

        completeness_multiple_thr[f"{distance_thr}mm"] = np.mean(completeness_count > 0)
        accuracy_multiple_thr[f"{distance_thr}mm"] = np.mean(accuracy_count > 0)
        # import ipdb; ipdb.set_trace()

    return accuracy_multiple_thr, completeness_multiple_thr


if __name__ == "__main__":
    args = parse_args()
    database_path = args.data_base_path
    sfm_base_path = args.sfm_base_path
    assert osp.exists(database_path) and osp.exists(sfm_base_path)

    object_names = os.listdir(database_path)
    id2full_name = {name[:4]: name for name in object_names if "-" in name}

    gathered_accuracy_metrics = {}
    gathered_completness_metrics = {}

    obj_ids = args.obj_ids
    for obj_id in tqdm(obj_ids):
        if obj_id in id2full_name:
            obj_full_name = id2full_name[obj_id]
            gt_mesh_path = osp.join(database_path, obj_full_name, "model.ply")
            sfm_pointcloud_path = osp.join(
                sfm_base_path, obj_full_name, "box_filter.ply"
            )
            accuracy_dict, completeness_dict = compute_pointcloud_accu_and_complete(
                gt_mesh_path,
                sfm_pointcloud_path,
                thrs=args.threshold,
                model_unit=args.model_unit,
            )

            for metric_name, metric in accuracy_dict.items():
                if metric_name not in gathered_accuracy_metrics:
                    gathered_accuracy_metrics[metric_name] = [metric]
                else:
                    gathered_accuracy_metrics[metric_name].append(metric)

            for metric_name, metric in completeness_dict.items():
                if metric_name not in gathered_completness_metrics:
                    gathered_completness_metrics[metric_name] = [metric]
                else:
                    gathered_completness_metrics[metric_name].append(metric)
        else:
            logger.warning(f"Obj id:{obj_id} not exists in data!")

    # Average on all objects.
    print("****** accuracy ****** \n")
    for metric_name, metric in gathered_accuracy_metrics.items():
        print(f"{metric_name}:")
        # metric_parsed = pd.DataFrame(metric)
        # print(metric_parsed.describe())
        metric_np = np.array(metric)
        metric_mean = np.mean(metric)
        print(metric_mean)
        print("---------------------\n")

    print("****** completeness ****** \n")
    for metric_name, metric in gathered_completness_metrics.items():
        print(f"{metric_name}:")
        # metric_parsed = pd.DataFrame(metric)
        # print(metric_parsed.describe())
        metric_np = np.array(metric)
        metric_mean = np.mean(metric)
        print(metric_mean)
        print("---------------------\n")

