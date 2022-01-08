from copy import deepcopy
import json
import os
import os.path as osp

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
import glob
import hydra
import math
import ray

import open3d as o3d
import os.path as osp
from tqdm import tqdm

from loguru import logger
from pathlib import Path
from matplotlib.pyplot import box
from omegaconf import DictConfig

from src.utils.ray_utils import ProgressBar, chunks


def parseScanData(cfg):
    """ Parse arkit scanning data"""
    # TODO: add arkit data processing
    pass


# def merge_(
#     anno_2d_file,
#     avg_anno_3d_file,
#     collect_anno_3d_file,
#     idxs_file,
#     img_id,
#     ann_id,
#     images,
#     annotations,
# ):
#     """ Merge annotations about difference objs"""
#     with open(anno_2d_file, "r") as f:
#         annos_2d = json.load(f)

#     for anno_2d in annos_2d:
#         img_id += 1
#         info = {
#             "id": img_id,
#             "img_file": anno_2d["img_file"],
#         }
#         images.append(info)

#         ann_id += 1
#         anno = {
#             "image_id": img_id,
#             "id": ann_id,
#             "pose_file": anno_2d["pose_file"],
#             "anno2d_file": anno_2d["anno_file"],
#             "avg_anno3d_file": avg_anno_3d_file,
#             "collect_anno3d_file": collect_anno_3d_file,
#             "idxs_file": idxs_file,
#         }
#         annotations.append(anno)
#     return img_id, ann_id


# def merge_anno(cfg):
#     """ Merge different objects' anno file into one anno file """
#     anno_dirs = []
#     names = cfg.names

#     if isinstance(names, str):
#         # Parse object directory
#         assert isinstance(names, str)
#         exception_obj_name_list = cfg.exception_obj_names
#         top_k_obj = cfg.top_k_obj
#         logger.info(f"Process all objects in directory:{names}")

#         object_names = []
#         object_names_list = os.listdir(names)[:top_k_obj]
#         for object_name in object_names_list:
#             if "-" not in object_name:
#                 continue
#             if object_name in exception_obj_name_list:
#                 continue
#             object_names.append(object_name)

#         names = object_names

#     all_data_names = os.listdir(
#         osp.join(
#             cfg.datamodule.data_dir,
#             f"outputs_{cfg.match_type}_{cfg.network.detection}_{cfg.network.matching}",
#         )
#     )
#     id2datafullname = {
#         data_name[:4]: data_name for data_name in all_data_names if "-" in data_name
#     }
#     for name in names:
#         if len(name) == 4:
#             # ID only!
#             if name in id2datafullname:
#                 name = id2datafullname[name]
#             else:
#                 logger.warning(f'id {name} not exist in sfm directory')
#         anno_dir = osp.join(
#             cfg.datamodule.data_dir,
#             f"outputs_{cfg.match_type}_{cfg.network.detection}_{cfg.network.matching}",
#             name,
#             "anno",
#         )
#         anno_dirs.append(anno_dir)

#     img_id = 0
#     ann_id = 0
#     images = []
#     annotations = []

#     for anno_dir in tqdm(anno_dirs):
#         logger.info(f"Merging anno dir: {anno_dir}")
#         anno_2d_file = osp.join(anno_dir, "anno_2d.json")
#         avg_anno_3d_file = osp.join(anno_dir, "anno_3d_average.npz")
#         collect_anno_3d_file = osp.join(anno_dir, "anno_3d_collect.npz")
#         idxs_file = osp.join(anno_dir, "idxs.npy")

#         if (
#             not osp.isfile(anno_2d_file)
#             or not osp.isfile(avg_anno_3d_file)
#             or not osp.isfile(collect_anno_3d_file)
#         ):
#             logger.info(f"No annotation in: {anno_dir}")
#             continue

#         img_id, ann_id = merge_(
#             anno_2d_file,
#             avg_anno_3d_file,
#             collect_anno_3d_file,
#             idxs_file,
#             img_id,
#             ann_id,
#             images,
#             annotations,
#         )

#     logger.info(f"Total num: {len(images)}")
#     instance = {"images": images, "annotations": annotations}

#     out_dir = osp.dirname(cfg.datamodule.out_path)
#     Path(out_dir).mkdir(exist_ok=True, parents=True)
#     with open(cfg.datamodule.out_path, "w") as f:
#         json.dump(instance, f)


def sfm(cfg):
    """ Sparse reconstruction and postprocess (on 3d points and features)"""
    data_dirs = cfg.dataset.data_dir

    if isinstance(data_dirs, str):
        # Parse object directory
        # assert isinstance(data_dirs, str)
        num_seq = cfg.dataset.num_seq
        exception_obj_name_list = cfg.dataset.exception_obj_names
        top_k_obj = cfg.dataset.top_k_obj
        if num_seq is not None:
            assert num_seq > 0
        logger.info(
            f"Process all objects in directory:{data_dirs}, process: {num_seq if num_seq is not None else 'all'} sequences"
        )

        object_names = os.listdir(data_dirs)[:top_k_obj]
        data_dirs_list = []

        if cfg.dataset.ids is not None:
            # Use data ids
            id2full_name = {name[:4]: name for name in object_names if '-' in name}
            object_names = [id2full_name[id] for id in cfg.dataset.ids if id in id2full_name]

        for object_name in object_names:
            if "-" not in object_name:
                continue

            if object_name in exception_obj_name_list:
                continue
            sequence_names = sorted(os.listdir(osp.join(data_dirs, object_name)))
            sequence_names = [
                sequence_name
                for sequence_name in sequence_names
                if "-" in sequence_name
            ][:num_seq]
            data_dirs_list.append(
                " ".join([osp.join(data_dirs, object_name)] + sequence_names)
            )

        data_dirs = data_dirs_list

    if not cfg.use_global_ray:
        sfm_worker(data_dirs, cfg)
    else:
        # Init ray
        if cfg.ray.slurm:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg.ray.n_workers * cfg.ray.n_cpus_per_worker),
                num_gpus=math.ceil(cfg.ray.n_workers * cfg.ray.n_gpus_per_worker),
                local_mode=cfg.ray.local_mode,
                ignore_reinit_error=True,
            )
        logger.info(f"Use ray for SfM mapping, total: {cfg.ray.n_workers} workers")

        pb = ProgressBar(len(data_dirs), "SfM Mapping begin...")
        all_subsets = chunks(data_dirs, math.ceil(len(data_dirs) / cfg.ray.n_workers))
        sfm_worker_results = [
            sfm_worker_ray_wrapper.remote(subset_data_dirs, cfg, pba=pb.actor)
            for subset_data_dirs in all_subsets
        ]
        pb.print_until_done()
        results = ray.get(sfm_worker_results)


def sfm_worker(data_dirs, cfg, pba=None):
    logger.info(f"Worker will process: {data_dirs}")
    data_dirs = tqdm(data_dirs) if pba is None else data_dirs
    for data_dir in data_dirs:
        logger.info(f"Processing {data_dir}.")
        root_dir, sub_dirs = data_dir.split(" ")[0], data_dir.split(" ")[1:]

        img_lists = []
        for sub_dir in sub_dirs:
            seq_dir = osp.join(root_dir, sub_dir)
            img_lists += glob.glob(
                str(Path(seq_dir)) + "/color_crop/*.png", recursive=True
            )

        # ------------------ downsample ------------------
        down_img_lists = []
        down_ratio = 5
        # down_ratio = 20
        for img_file in img_lists:
            index = int(img_file.split("/")[-1].split(".")[0])
            if index % down_ratio == 0:
                down_img_lists.append(img_file)
        img_lists = down_img_lists
        # -------------------------------------------------

        if len(img_lists) == 0:
            logger.info(f"No png image in {root_dir}")
            if pba is not None:
                pba.update.remote(1)
            continue

        obj_name = root_dir.split("/")[-1]
        outputs_dir_root = cfg.dataset.outputs_dir

        sfm_core(cfg, img_lists, outputs_dir_root, obj_name)
        postprocess(cfg, img_lists, root_dir, sub_dirs, outputs_dir_root, obj_name)
        if pba is not None:
            pba.update.remote(1)
    return None


@ray.remote(num_cpus=1)  # release gpu after finishing
def sfm_worker_ray_wrapper(data_dirs, cfg, pba=None):
    return sfm_worker(data_dirs, cfg, pba)


def sfm_core(cfg, img_lists, outputs_dir_root, obj_name):
    """ Sparse reconstruction: extract features, match features, triangulation
                            outputs_dir_root
                                   |
                           ------------------
                           |
                    method_names...
                        |
                -------------------
                |                 |
            obj_names....        vis3d
                                    |
                            -----------
                            |         
                        ojb_names...
    """
    from src.hloc import (
        extract_features,
        pairs_from_covisibility,
        match_features,
        generate_empty,
        triangulation,
        pairs_from_poses,
    )
    from src.NeuralSfM import coarse_match, post_optimization

    outputs_dir = osp.join(
        outputs_dir_root,
        "outputs_"
        + cfg.match_type
        + "_"
        + cfg.network.detection
        + "_"
        + cfg.network.matching,
        obj_name,
    )
    vis3d_pth = osp.join(
        outputs_dir_root,
        "outputs_"
        + cfg.match_type
        + "_"
        + cfg.network.detection
        + "_"
        + cfg.network.matching,
        "vis3d",
        obj_name,
    )

    feature_out = osp.join(outputs_dir, f"feats-{cfg.network.detection}.h5")
    covis_num = cfg.sfm.covis_num
    covis_pairs_out = osp.join(outputs_dir, f"pairs-covis{covis_num}.txt")
    matches_out = osp.join(outputs_dir, f"matches-{cfg.network.matching}.h5")
    empty_dir = osp.join(outputs_dir, "sfm_empty")
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")
    visualize_dir = osp.join(outputs_dir, "visualize")

    if cfg.redo:
        if cfg.network.matching != "loftr":
            os.system(f"rm -rf {outputs_dir}")
            Path(outputs_dir).mkdir(exist_ok=True, parents=True)

            extract_features.main(img_lists, feature_out, cfg)
            # pairs_from_covisibility.covis_from_index(img_lists, covis_pairs_out, num_matched=covis, gap=cfg.sfm.gap)
            pairs_from_poses.covis_from_pose(
                img_lists,
                covis_pairs_out,
                covis_num,
                max_rotation=cfg.sfm.rotation_thresh,
            )
            match_features.main(
                cfg, feature_out, covis_pairs_out, matches_out, vis_match=False
            )
            generate_empty.generate_model(img_lists, empty_dir)
            if cfg.use_global_ray:
                # Need to ask for gpus!
                triangulation_results = triangulation.main_ray_wrapper.remote(
                    deep_sfm_dir,
                    empty_dir,
                    outputs_dir,
                    covis_pairs_out,
                    feature_out,
                    matches_out,
                    match_model=cfg.network.matching,
                    image_dir=None,
                )
                results = ray.get(triangulation_results)
            else:
                triangulation.main(
                    deep_sfm_dir,
                    empty_dir,
                    outputs_dir,
                    covis_pairs_out,
                    feature_out,
                    matches_out,
                    match_model=cfg.network.matching,
                    image_dir=None,
                )
            # torch.cuda.synchronize()
        else:
            if cfg.overwrite_all:
                os.system(f"rm -rf {outputs_dir}")
                os.system(f"rm -rf {vis3d_pth}")
            Path(outputs_dir).mkdir(exist_ok=True, parents=True)

            if (
                not osp.exists(osp.join(deep_sfm_dir, "model_coarse"))
                or cfg.overwrite_coarse
            ):
                logger.info("LoFTR coarse mapping begin...")
                os.system(f"rm -rf {empty_dir}")
                os.system(f"rm -rf {deep_sfm_dir}")
                os.system(
                    f'rm -rf {osp.join(covis_pairs_out.rsplit("/", 1)[0], "fine_matches.pkl")}'
                )  # Force refinement to recompute fine match

                pairs_from_covisibility.covis_from_index(
                    img_lists, covis_pairs_out, num_matched=covis_num, gap=cfg.sfm.gap
                )
                coarse_match.loftr_coarse_matching(
                    img_lists,
                    covis_pairs_out,
                    feature_out,
                    matches_out,
                    use_ray=cfg.use_local_ray,
                )
                generate_empty.generate_model(img_lists, empty_dir)
                if cfg.use_global_ray:
                    # Need to ask for gpus!
                    triangulation_results = triangulation.main_ray_wrapper.remote(
                        deep_sfm_dir,
                        empty_dir,
                        outputs_dir,
                        covis_pairs_out,
                        feature_out,
                        matches_out,
                        match_model=cfg.network.matching,
                        image_dir=None,
                    )
                    results = ray.get(triangulation_results)
                else:
                    triangulation.main(
                        deep_sfm_dir,
                        empty_dir,
                        outputs_dir,
                        covis_pairs_out,
                        feature_out,
                        matches_out,
                        match_model=cfg.network.matching,
                        image_dir=None,
                    )

                if cfg.enable_loftr_post_refine:
                    assert osp.exists(osp.join(deep_sfm_dir, "model"))
                    os.system(
                        f"mv {osp.join(deep_sfm_dir, 'model')} {osp.join(deep_sfm_dir, 'model_coarse')}"
                    )
                    os.system(
                        f"mv {feature_out} {osp.splitext(feature_out)[0] + '_coarse' + osp.splitext(feature_out)[1]}"
                    )

            if cfg.enable_loftr_post_refine:
                if (
                    not osp.exists(osp.join(deep_sfm_dir, "model"))
                    or cfg.overwrite_fine
                ):
                    assert osp.exists(
                        osp.join(deep_sfm_dir, "model_coarse")
                    ), f"model_coarse not exist under: {deep_sfm_dir}, please set 'cfg.overwrite_coarse = True'"
                    os.system(f"rm -rf {osp.join(deep_sfm_dir, 'model')}")

                    logger.info("LoFTR mapping post refinement begin...")
                    state, _, _ = post_optimization.post_optimization(
                        img_lists,
                        covis_pairs_out,
                        colmap_coarse_dir=osp.join(deep_sfm_dir, "model_coarse"),
                        refined_model_save_dir=osp.join(deep_sfm_dir, "model"),
                        match_out_pth=matches_out,
                        feature_out_pth=feature_out,
                        use_global_ray=cfg.use_global_ray,
                        fine_match_use_ray=cfg.use_local_ray,
                        visualize_dir=visualize_dir,
                        vis3d_pth=vis3d_pth,
                    )
                    if state == False:
                        logger.error("colmap coarse is empty!")


def postprocess(cfg, img_lists, root_dir, sub_dirs, outputs_dir_root, obj_name):
    """ Filter points and average feature"""
    from src.hloc.postprocess import filter_points, feature_process, filter_tkl

    data_dir0 = osp.join(root_dir, sub_dirs[0])
    # bbox_path = osp.join(data_dir0, 'RefinedBox.txt')
    # bbox_path = bbox_path if osp.isfile(bbox_path) else osp.join(data_dir0, 'Box.txt')
    bbox_path = osp.join(data_dir0, "Box.txt")
    trans_box_path = osp.join(data_dir0, "Box_trans.txt")

    match_type = cfg.match_type
    outputs_dir = osp.join(
        outputs_dir_root,
        "outputs_"
        + cfg.match_type
        + "_"
        + cfg.network.detection
        + "_"
        + cfg.network.matching,
        obj_name,
    )
    vis3d_pth = osp.join(
        outputs_dir_root,
        "outputs_"
        + cfg.match_type
        + "_"
        + cfg.network.detection
        + "_"
        + cfg.network.matching,
        "vis3d",
        obj_name,
    )
    feature_out = osp.join(outputs_dir, f"feats-{cfg.network.detection}.h5")
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")
    model_path = osp.join(deep_sfm_dir, "model")

    # select track length to limit the number of 3d points below thres.
    track_length, points_count_list = filter_tkl.get_tkl(
        model_path, thres=cfg.dataset.max_num_kp3d, show=False
    )
    tkl_file_path = filter_tkl.vis_tkl_filtered_pcds(
        model_path, points_count_list, track_length, outputs_dir, vis3d_pth
    )  # visualization only

    xyzs, points_ids = filter_points.filter_3d(
        model_path, track_length, bbox_path, box_trans_path=trans_box_path
    )  # crop 3d points by 3d box and track length
    merge_xyzs, merge_idxs = filter_points.merge(
        xyzs, points_ids, dist_threshold=1e-3
    )  # merge 3d points by distance between points
    if cfg.debug:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merge_xyzs)
        out_file = osp.join(outputs_dir, "box_filter.ply")
        o3d.io.write_point_cloud(out_file, pcd)

    # Save loftr coarse keypoints:
    cfg_dup = deepcopy(cfg)
    cfg_dup.network.detection = "loftr_coarse"
    feature_coarse_path = (
        osp.splitext(feature_out)[0] + "_coarse" + osp.splitext(feature_out)[1]
    )
    # FIXME: bug here!
    feature_process.get_kpt_ann(
        cfg_dup,
        img_lists,
        feature_coarse_path,
        outputs_dir,
        merge_idxs,
        merge_xyzs,
        save_feature_for_each_image=False,
    )

    feature_process.get_kpt_ann(
        cfg,
        img_lists,
        feature_out,
        outputs_dir,
        merge_idxs,
        merge_xyzs,
        save_feature_for_each_image=False,
    )


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
