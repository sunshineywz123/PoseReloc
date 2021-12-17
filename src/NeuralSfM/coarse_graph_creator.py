# 1. Build dataset
# 2. LoFTR coarse matching and saving
# 3. Convert to COLMAP friendly format
# 4. Run COLMAP and save results
import argparse
from typing import ChainMap
import ray
import os
import os.path as osp

from ray import state

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
import math
import pytorch_lightning as pl
import torch
import numpy as np
from ray.actor import ActorHandle
from loguru import logger
from shutil import copyfile, rmtree
import subprocess
import multiprocessing
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from time import time
from joblib import Parallel, delayed
from tqdm import tqdm

from datasets.loftr_dataset import IMCValPairDataset
from src.utils.ray_utils import ProgressBar
from src.neural_sfm.matcher_model.utils import chunk_index

from src.config.default import get_cfg_defaults
from src.matcher_loftr import Matcher_LoFTR as LoFTR_SfM
from src.utils.misc import lower_config
from src.utils.torch_utils import update_state_dict, STATE_DICT_MAPPER
from src.extractors import build_extractor
from src.loftr_sfm.utils.detector_wrapper import DetectorWrapper, DetectorWrapperTwoView
from src.utils.utils import pose_auc
from test.imc.utils import (
    chunks,
    extract_geo_model_inliers,
    Match2Kpts,
    save_h5,
    load_h5,
    agg_groupby_2d,
    split_dict,
)
from third_party.colmap.scripts.python.read_write_model import read_images_binary
from tools.imc_utils import load_image, load_calib, compute_stereo_metrics_from_colmap

from src.extractors import build_extractor_for_spg
from src.matcher_spg import *


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Match and refinement model related parameters
    parser.add_argument("--cfg_path", type=str, default="")
    parser.add_argument("--weight_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--match_type",
        type=str,
        choices=["loftr_coarse", "SuperGlue"],
        default="loftr_coarse",
    )

    # Date related
    parser.add_argument("--img_resize_max", type=int, default=1920)
    parser.add_argument("--img_resize_min", type=int, default=800)
    parser.add_argument("--df", type=int, default=8, help="divisible factor.")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="IMC",
        choices=["IMC"],
        help="Choice: [IMC,]",
    )
    parser.add_argument(
        "--subset_name",
        default=None,
    )
    parser.add_argument(
        "--data_part_name",
        default=None,
        help="Used to identify different dataset results",
    )
    parser.add_argument("--n_imgs", default=None, help="Used for debug")

    # Raw match cache related
    parser.add_argument(
        "--cache", action="store_true", help="cache matching results for debugging."
    )

    # Results save configs
    parser.add_argument(
        "--save_dir", type=str, default="/data/hexingyi/NeuralSfM_results"
    )

    # Ransac related
    parser.add_argument(
        "--inlier_only",
        action="store_true",
        help="only store ransac inliers and its corresponding kpts.",
    )
    parser.add_argument("--geo_model", type=str, choices=["F", "E"], default="F")
    parser.add_argument(
        "--ransac_method",
        type=str,
        choices=["RANSAC", "DEGENSAC", "MAGSAC"],
        default="DEGENSAC",
    )
    parser.add_argument("--pixel_thr", type=float, default=1.0)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--conf_thr", type=float, default=0.99999)

    # Ray related
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--n_cpus_per_worker", type=float, default=1)
    parser.add_argument("--n_gpus_per_worker", type=float, default=1)
    parser.add_argument(
        "--local_mode", action="store_true", help="ray local mode for debugging."
    )

    parser.add_argument("--match_debug", action="store_true")

    # COLMAP related
    parser.add_argument("--colmap_debug", action="store_true")
    parser.add_argument("--colmap_verbose", action="store_true")
    parser.add_argument(
        "--colmap_min_model_size",
        type=int,
        default=3,
        help="Minium size to be used for mapper",
    )
    parser.add_argument("--colmap_filter_max_reproj_error", type=int, default=4)

    # COLMAP visualization related:
    parser.add_argument(
        "--visual_enable",
        action="store_true",
        help="Whether run COLMAP results visualization",
    )
    parser.add_argument("--visual_best_index", type=int, default=0)

    # COLMAP eval related:
    parser.add_argument("--eval_best_index", type=int, default=0)

    args = parser.parse_args()

    # Post process of args
    base_dir_part = [args.save_dir]
    base_dir_part.append(
        args.data_part_name
    ) if args.data_part_name is not None else None
    base_dir_part.append(
        osp.splitext(osp.basename(args.subset_path))[0]
    ) if args.subset_name is not None else None
    base_dir_part.append(args.match_type)
    args.save_dir = osp.join(*base_dir_part)
    args.cache_dir = osp.join(*base_dir_part, "raw_matches")
    return args


def build_model(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_path)
    pl.seed_everything(args.seed)

    if args.match_type == "loftr_coarse":
        detector = build_extractor(lower_config(cfg.LOFTR_MATCH_FINE))
        detector = (
            DetectorWrapper(
                detector,
                cfg.LOFTR_MATCH_FINE.DETECTOR,
                fullcfg=lower_config(cfg.LOFTR_MATCH_FINE),
            )
            if not cfg.LOFTR_GUIDED_MATCHING.ENABLE
            else DetectorWrapperTwoView(
                detector,
                cfg.LOFTR_MATCH_FINE.DETECTOR,
                fullcfg=lower_config(cfg.LOFTR_MATCH_FINE),
            )
        )

        match_cfg = {
            "loftr_backbone": lower_config(cfg.LOFTR_BACKBONE),
            "loftr_coarse": lower_config(cfg.LOFTR_COARSE),
            "loftr_match_coarse": lower_config(cfg.LOFTR_MATCH_COARSE),
            "loftr_fine": lower_config(cfg.LOFTR_FINE),
            "loftr_match_fine": lower_config(cfg.LOFTR_MATCH_FINE),
            "loftr_guided_matching": lower_config(cfg.LOFTR_GUIDED_MATCHING),
        }
        matcher = LoFTR_SfM(config=match_cfg).eval()
        # load checkpoints
        state_dict = torch.load(args.weight_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k.replace("matcher.", "")] = state_dict.pop(k)
        try:
            matcher.load_state_dict(state_dict, strict=True)
        except RuntimeError as _:
            state_dict, updated = update_state_dict(
                STATE_DICT_MAPPER, state_dict=state_dict
            )
            assert updated
            matcher.load_state_dict(state_dict, strict=True)

    elif args.match_type == "SuperGlue":
        detector_type = cfg.SUPERGLUE.DETECTOR_TYPE

        assert detector_type in [
            "SuperPoint",
            "SVCNN",
            "Disk",
            "D2_net",
            "R2D2",
        ], "wrong detector name!"
        match_cfg = {
            "superglue": {k.lower(): v for k, v in cfg.SUPERGLUE.items()},
            "loss": {k.lower(): v for k, v in cfg.SUPERGLUE_LOSS.items()},
            "pose_estimation_method": cfg.TRAINER.POSE_ESTIMATION_METHOD,
            "ransac_pixel_thresh": cfg.TRAINER.RANSAC_PIXEL_THR,
            # "data_source": cfg.data_source,
            "mask_scale": cfg.LOFTR_BACKBONE.RESOLUTION[0],
            "detector_type": detector_type,
        }
        match_cfg["detector"] = {k.lower(): v for k, v in cfg.SUPERPOINT.items()}
        # match_cfg["detector"]["max_keypoints"] = args.n_kpts
        match_cfg["superglue"]["descriptor_dim"] = match_cfg["detector"][
            "descriptor_dim"
        ]

        # Detector SVCNN or Superpoint
        detector = build_extractor_for_spg(match_cfg)
        # Matcher
        matcher = SuperGlue(match_cfg.get("superglue", {}), profiler=None)

        # Load superglue checkpoints
        state_dict = torch.load(args.weight_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            temp = state_dict.pop(k)
            if "superglue" in k:
                state_dict[k.replace("superglue.", "")] = temp
        try:
            matcher.load_state_dict(state_dict, strict=True)
        except RuntimeError as _:
            state_dict, updated = update_state_dict(
                STATE_DICT_MAPPER, state_dict=state_dict
            )
            assert updated
            matcher.load_state_dict(state_dict, strict=True)
        args.cfg = cfg

        detector.eval()
        matcher.eval()
    else:
        raise NotImplementedError
    return detector, matcher


def extract_preds(data, args):
    """extract predictions assuming bs==1"""
    if args.match_type == "loftr_coarse":
        m_bids = data["m_bids"].cpu().numpy()
        assert (np.unique(m_bids) == 0).all()
        mkpts0 = data["mkpts0_f"].cpu().numpy()
        mkpts1 = data["mkpts1_f"].cpu().numpy()
        mconfs = data["mconf"].cpu().numpy()
    elif args.match_type == "SuperGlue":
        # TODO: This should be moved to extract matches and build an unifrom format
        if "scale0" in data:
            kpts0 = data["keypoints0"] * data["scale0"][:, [1, 0]]
            kpts0 = kpts0[0].cpu().numpy()
            kpts1 = data["keypoints1"] * data["scale1"][:, [1, 0]]
            kpts1 = kpts1[0].cpu().numpy()
        else:
            kpts0 = data["keypoints0"][0].cpu().numpy()
            kpts1 = data["keypoints1"][0].cpu().numpy()

        matches = data["matches0"][0].cpu().numpy()  # -1 means no matches!
        conf = data["matching_scores0"][0].cpu().numpy()
        valid = matches > -1
        if len(matches) == 0:
            matches = np.empty((0, 2))

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconfs = conf[valid]
    else:
        raise NotImplementedError

    detector_kpts_mask = (
        data["detector_kpts_mask"].cpu().numpy()
        if "detector_kpts_mask" in data
        else np.zeros_like(mconfs)
    )
    return mkpts0, mkpts1, mconfs, detector_kpts_mask


def extract_inliers(data, args):
    """extract inlier matches assume bs==1.
    NOTE: If no inliers found, keep all matches.
    """
    mkpts0, mkpts1, mconfs, detector_kpts_mask = extract_preds(data, args)
    K0 = data["K0"][0].cpu().numpy() if args.geo_model == "E" else None
    K1 = data["K1"][0].cpu().numpy() if args.geo_model == "E" else None
    if len(mkpts0) >= 8:
        inliers = extract_geo_model_inliers(
            mkpts0,
            mkpts1,
            mconfs,
            args.geo_model,
            args.ransac_method,
            args.pixel_thr,
            args.max_iters,
            args.conf_thr,
            K0=K0,
            K1=K1,
        )
        mkpts0, mkpts1, mconfs, detector_kpts_mask = map(
            lambda x: x[inliers], [mkpts0, mkpts1, mconfs, detector_kpts_mask]
        )
        # logger.info(f"total:{inliers.shape[0]}, {inliers.sum()} inliers, filtered: {inliers.shape[0] - inliers.sum()}")
    # assert mkpts0.shape[0] != 0
    return mkpts0, mkpts1, mconfs, detector_kpts_mask


@torch.no_grad()
def extract_matches(data, detector=None, matcher=None, args=None, inlier_only=True):
    # 1. inference
    if args.match_type == "loftr_coarse":
        detector(data)
        matcher(data)
    elif args.match_type == "SuperGlue":
        "Only support batch size = 1"
        pred = extract_features(detector, data, mode="eval")
        data = {**data, **pred}
        for k in data:
            if isinstance(
                data[k], (list, tuple)
            ):  # assume superpoint's predictions padded
                if isinstance(data[k][0], torch.Tensor):
                    data[k] = torch.stack(data[k])
        assign_matrix = matcher(data)
        match_result = matcher.post_process(None, data, assign_matrix)
        data = {**data, **match_result}

    # 2. run RANSAC and extract inliers
    mkpts0, mkpts1, mconfs, detector_kpts_mask = (
        extract_inliers(data, args) if inlier_only else extract_preds(data, args)
    )
    del data
    torch.cuda.empty_cache()
    return (mkpts0, mkpts1, mconfs, detector_kpts_mask)


@ray.remote(num_cpus=1, num_gpus=1, max_calls=1)  # release gpu after finishing
@torch.no_grad()
def match_worker(dataset, subset_ids, args, pba: ActorHandle):
    """extract matches from part of the possible image pair permutations"""
    detector, matcher = build_model(args)
    detector.cuda()
    matcher.cuda()
    matches = {}
    # match all permutations
    for subset_id in subset_ids:
        data = dataset[subset_id]
        f_name0, f_name1 = data["f_name0"], data["f_name1"]
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        mkpts0, mkpts1, mconfs, detector_kpts_mask = extract_matches(
            data_c,
            detector=detector,
            matcher=matcher,
            args=args,
            inlier_only=args.inlier_only,
        )

        # Extract matches (kpts-pairs & scores)
        matches["-".join([f_name0, f_name1])] = np.concatenate(
            [mkpts0, mkpts1, mconfs[:, None]], -1
        )  # (N, 5)
        pba.update.remote(1)
    return matches


@ray.remote(num_cpus=1)
def keypoint_worker(name_kpts, args, pba: ActorHandle):
    """merge keypoints associated with one image.
    python >= 3.7 only.
    """
    keypoints = {}
    for name, kpts in name_kpts:
        # filtering
        kpt2score = agg_groupby_2d(kpts[:, :2].astype(int), kpts[:, -1], agg="sum")
        kpt2id_score = {
            k: (i, v)
            for i, (k, v) in enumerate(
                sorted(kpt2score.items(), key=lambda kv: kv[1], reverse=True)
            )
        }
        keypoints[name] = kpt2id_score

        pba.update.remote(1)
    return keypoints


@ray.remote(num_cpus=1)
def update_matches(matches, keypoints, args, pba: ActorHandle):
    # convert match to indices
    ret_matches = {}

    for k, v in matches.items():
        mkpts0, mkpts1 = (
            map(tuple, v[:, :2].astype(int)),
            map(tuple, v[:, 2:4].astype(int)),
        )
        name0, name1 = k.split("-")
        _kpts0, _kpts1 = keypoints[name0], keypoints[name1]

        mids = np.array(
            [
                [_kpts0[p0][0], _kpts1[p1][0]]
                for p0, p1 in zip(mkpts0, mkpts1)
                if p0 in _kpts0 and p1 in _kpts1
            ]
        )

        assert (
            len(mids) == v.shape[0]
        ), f"len mids: {len(mids)}, num matches: {v.shape[0]}"
        if len(mids) == 0:
            mids = np.empty((0, 2))

        ret_matches[k] = mids.T.astype(int)  # (2, N) - IMC submission format
        pba.update.remote(1)

    return ret_matches


@ray.remote(num_cpus=1)
def transform_keypoints(keypoints, args, pba: ActorHandle):
    """assume keypoints sorted w.r.t. score"""
    ret_kpts = {}
    ret_scores = {}
    for k, v in keypoints.items():
        v = {_k: _v for _k, _v in v.items() if len(_k) == 2}
        kpts = np.array([list(kpt) for kpt in v.keys()]).astype(np.float32)
        scores = np.array([s[-1] for s in v.values()]).astype(np.float32)
        assert len(kpts) != 0, "corner-case n_kpts=0 not handled."
        ret_kpts[k] = kpts
        ret_scores[k] = scores
        pba.update.remote(1)
    return ret_kpts, ret_scores


def exhaustive_matching(args, dataset):
    # Matcher runner
    if args.cache and osp.exists(osp.join(args.cache_dir, "raw_matches.h5")):
        matches = load_h5(
            osp.join(args.cache_dir, "raw_matches.h5"), transform_slash=True
        )
        logger.info("Caches raw matches loaded!")
    else:
        pb = ProgressBar(len(dataset), "Matching image pairs...")
        all_subset_ids = chunk_index(
            len(dataset), math.ceil(len(dataset) / args.n_workers)
        )
        obj_refs = [
            match_worker.remote(dataset, subset_ids, args, pb.actor)
            for subset_ids in all_subset_ids
        ]
        pb.print_until_done()
        results = ray.get(obj_refs)
        matches = dict(ChainMap(*results))
        logger.info("Matcher finish!")

        # over write anyway
        os.makedirs(args.cache_dir, exist_ok=True)
        save_h5(matches, osp.join(args.cache_dir, "raw_matches.h5"))
        logger.info(f"Raw matches cached: {args.cache_dir}")

    # Combine keypoints
    n_imgs = len(dataset.f_names)
    pb = ProgressBar(n_imgs, "Combine keypoints")
    all_kpts = Match2Kpts(matches, dataset.f_names)
    sub_kpts = chunks(all_kpts, math.ceil(n_imgs / args.n_workers))
    obj_refs = [keypoint_worker.remote(sub_kpt, args, pb.actor) for sub_kpt in sub_kpts]
    pb.print_until_done()
    keypoints = dict(ChainMap(*ray.get(obj_refs)))
    logger.info("Combine keypoints finish!")

    # Convert keypoints match to keypoints indexs
    pb = ProgressBar(len(matches), "Updating matches...")
    _keypoints_ref = ray.put(keypoints)
    obj_refs = [
        update_matches.remote(sub_matches, _keypoints_ref, args, pb.actor)
        for sub_matches in split_dict(matches, math.ceil(len(matches) / args.n_workers))
    ]
    pb.print_until_done()
    updated_matches = dict(ChainMap(*ray.get(obj_refs)))

    # Post process keypoints:
    keypoints = {
        k: v for k, v in keypoints.items() if isinstance(v, dict)
    }  # assume filename in f'{xxx}_{yyy}' format
    pb = ProgressBar(len(keypoints), "Post-processing keypoints...")
    obj_refs = [
        transform_keypoints.remote(sub_kpts, args, pb.actor)
        for sub_kpts in split_dict(
            keypoints, math.ceil(len(keypoints) / args.n_workers)
        )
    ]
    pb.print_until_done()
    kpts_scores = ray.get(obj_refs)
    final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
    final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    # Save finial results:
    os.makedirs(args.save_dir, exist_ok=True)
    save_h5(final_keypoints, osp.join(args.save_dir, "keypoints.h5"))
    save_h5(updated_matches, osp.join(args.save_dir, "matches.h5"))

    return final_keypoints, updated_matches


def colmapRunner(dataset, keypoints_dict, matches_dict, args):
    # Build Colmap file path
    base_path = args.save_dir
    os.makedirs(base_path, exist_ok=True)
    colmap_temp_path = osp.join(base_path, "colmap_temp_path")
    colmap_output_path = osp.join(base_path, "colmap_output_path")
    # create temp directory
    if osp.exists(colmap_temp_path):
        logger.info(" -- temp path exists - cleaning up from crash")
        rmtree(colmap_temp_path)
        if os.path.exists(colmap_output_path):
            rmtree(colmap_output_path)

    # create output directory
    if osp.exists(colmap_output_path):
        if not args.colmap_debug:
            logger.info("colmap results already exists, don't need to run colmap")
            return
        else:
            rmtree(colmap_output_path)

    os.makedirs(colmap_temp_path)
    os.makedirs(colmap_output_path)

    # Create colmap-friendy structures
    os.makedirs(os.path.join(colmap_temp_path, "images"))
    os.makedirs(os.path.join(colmap_temp_path, "features"))
    img_paths = dataset.img_paths
    pair_ids = dataset.pair_ids

    # TODO: add use visible 

    # copy images
    for _src in img_paths:
        _dst = osp.join(colmap_temp_path, "images", osp.basename(_src))
        copyfile(_src, _dst)
    logger.info(f"Image copy finish! Copy {len(img_paths)} images!")

    num_kpts = []
    # write features to colmap friendly format
    for img_path in img_paths:
        img_name = osp.basename(img_path)
        f_name = osp.splitext(img_name)[0]
        # load keypoints:
        keypoints = keypoints_dict[f_name]
        # kpts file to write to:
        kp_file = osp.join(colmap_temp_path, "features", img_name + ".txt")
        num_kpts.append(keypoints.shape[0])
        # open file to write
        with open(kp_file, "w") as f:
            # Retieve the number of keypoints
            len_keypoints = len(keypoints)
            f.write(str(len_keypoints) + " " + str(128) + "\n")
            for i in range(len_keypoints):
                kp = " ".join(str(k) for k in keypoints[i][:4])
                desc = " ".join(str(0) for d in range(128))
                f.write(kp + " " + desc + "\n")
    logger.info(
        f"Feature format convert finish! Converted {len(img_paths)} images, have: {np.array(num_kpts)} keypoints"
    )

    # write matches to colmap friendly format
    match_file = os.path.join(colmap_temp_path, "matches.txt")
    num_matches = []
    with open(match_file, "w") as f:
        for pair_id in pair_ids:
            img0_name = os.path.basename(img_paths[pair_id[0]])
            img1_name = os.path.basename(img_paths[pair_id[1]])
            f0_name = os.path.splitext(img0_name)[0]
            f1_name = os.path.splitext(img1_name)[0]

            # Load matches
            key = "-".join([f0_name, f1_name])
            matches = np.squeeze(matches_dict[key])
            num_matches.append(matches.shape[1])
            # only write when matches are given
            if matches.ndim == 2:
                f.write(img0_name + " " + img1_name + "\n")
                for _i in range(matches.shape[1]):
                    f.write(str(matches[0, _i]) + " " + str(matches[1, _i]) + "\n")
                f.write("\n")
    logger.info(
        f"Match format convert finish, Converted {len(pair_ids)} pairs, min match: {np.array(num_matches).min()}, max match: {np.array(num_matches).max()}"
    )

    # COLMAP runs -- wrapped in try except to throw errors if subprocess fails
    # and then clean up the colmap temp directory

    try:
        print("COLMAP Feature Import")
        cmd = ["colmap", "feature_importer"]
        cmd += ["--database_path", os.path.join(colmap_output_path, "databases.db")]
        cmd += ["--image_path", os.path.join(colmap_temp_path, "images")]
        cmd += ["--import_path", os.path.join(colmap_temp_path, "features")]
        colmap_res = subprocess.run(cmd)

        if colmap_res.returncode != 0:
            raise RuntimeError(" -- COLMAP failed to import features!")

        print("COLMAP Match Import")
        cmd = ["colmap", "matches_importer"]
        cmd += ["--database_path", os.path.join(colmap_output_path, "databases.db")]
        cmd += ["--match_list_path", os.path.join(colmap_temp_path, "matches.txt")]
        cmd += ["--match_type", "raw"]
        cmd += ["--SiftMatching.use_gpu", "0"]
        colmap_res = subprocess.run(cmd)
        if colmap_res.returncode != 0:
            raise RuntimeError(" -- COLMAP failed to import matches!")

        print("COLMAP Mapper")
        cmd = ["colmap", "mapper"]
        cmd += ["--image_path", os.path.join(colmap_temp_path, "images")]
        cmd += ["--database_path", os.path.join(colmap_output_path, "databases.db")]
        cmd += ["--output_path", colmap_output_path]
        cmd += ["--Mapper.min_model_size", str(args.colmap_min_model_size)]

        # TODO: test multi threads:
        cmd += ["--Mapper.num_threads", str(min(multiprocessing.cpu_count(), 32))]

        # cmd += ["2>&1", "|", "tee", os.path.join(colmap_output_path, "output.txt")]

        cmd += [
            "--Mapper.filter_max_reproj_error",
            str(args.colmap_filter_max_reproj_error),
        ]

        if args.colmap_verbose:
            colmap_res = subprocess.run(cmd)
        else:
            colmap_res = subprocess.run(cmd, capture_output=True)
            with open(osp.join(colmap_output_path, "output.txt"), "w") as f:
                f.write(colmap_res.stdout.decode())

        if colmap_res.returncode != 0:
            raise RuntimeError(" -- COLMAP failed to run mapper!")
            # print("warning! colmap failed to run mapper!")

        # Delete temp directory after working
        rmtree(colmap_temp_path)

    except Exception as err:
        # Remove colmap output path and temp path
        rmtree(colmap_temp_path)
        rmtree(colmap_output_path)

        # Re-throw error
        print(err)
        raise RuntimeError("Parts of colmap runs returns failed state!")
    """
    # Check validity of colmap reconstruction for all of them
    is_any_colmap_valid = False
    idx_list = [
        os.path.join(colmap_output_path, _d)
        for _d in os.listdir(colmap_output_path)
        if os.path.isdir(os.path.join(colmap_output_path, _d))
    ]
    for idx in idx_list:
        colmap_img_file = os.path.join(idx, "images.bin")
        if is_colmap_img_valid(colmap_img_file):
            is_any_colmap_valid = True
            break
    if not is_any_colmap_valid:
        print("Error in reading colmap output -- " "removing colmap output directory")
        rmtree(colmap_output_path)
    """


def visualize_colmap(dataset, keypoints_dict, args):
    """Visualization of colmap points.

    Parameters
    ----------
    cfg: Namespace
        Configurations for running this part of the code.

    """
    base_path = args.save_dir
    assert osp.exists(base_path)
    t_start = time()

    # Create results folder if it does not exist
    viz_folder_hq = osp.join(base_path, "colmap_visualize", "png")
    viz_folder_lq = osp.join(base_path, "colmap_visualize", "jpg")
    if not os.path.exists(viz_folder_hq):
        os.makedirs(viz_folder_hq)
    if not os.path.exists(viz_folder_lq):
        os.makedirs(viz_folder_lq)

    # Get list of all images in this bag
    img_paths = dataset.img_paths
    pair_ids = dataset.pair_ids

    # Retrieve reconstruction
    colmap_output_path = osp.join(base_path, "colmap_output_path")
    assert osp.exists(
        colmap_output_path
    ), "Colmap output path not exists! Please check!"
    # is_colmap_valid = os.path.exists(
    #     os.path.join(colmap_output_path, '0'))
    best_index = args.visual_best_index
    if best_index != -1:
        colmap_images = read_images_binary(
            os.path.join(colmap_output_path, str(best_index), "images.bin")
        )
    registed_index = []
    registed_points_mask = []

    # Parser results
    subprocess.call(
        [
            "colmap",
            "model_converter",
            "--input_path",
            os.path.join(colmap_output_path, str(best_index)),
            "--output_path",
            os.path.join(colmap_output_path, str(best_index)),
            "--output_type",
            "TXT",
        ]
    )
    colmap_res = subprocess.run(
        [
            "colmap",
            "model_analyzer",
            "--path",
            os.path.join(colmap_output_path, str(best_index)),
        ],
        capture_output=True,
    )

    with open(osp.join(colmap_output_path, "output.txt"), "a") as f:
        f.write(colmap_res.stdout.decode())

    for i, image_path in enumerate(img_paths):
        # Load image and keypoints
        im, _ = load_image(
            image_path, use_color_image=True, crop_center=False, force_rgb=True
        )
        used = None
        key = os.path.splitext(os.path.basename(image_path))[0]
        if best_index != -1:
            for j in colmap_images:
                if key in colmap_images[j].name:
                    # plot all keypoints
                    used = colmap_images[j].point3D_ids != -1
                    registed_points_mask.append(used)
                    registed_index.append(i)
                    break
        if used is None:
            used = [False] * keypoints_dict[key].shape[0]
        used = np.array(used)

        fig = plt.figure(figsize=(20, 20))
        plt.imshow(im)
        plt.plot(
            keypoints_dict[key][~used, 0],
            keypoints_dict[key][~used, 1],
            "r.",
            markersize=12,
        )
        plt.plot(
            keypoints_dict[key][used, 0],
            keypoints_dict[key][used, 1],
            "b.",
            markersize=12,
        )
        plt.tight_layout()
        plt.axis("off")

        # TODO Ideally we would save to pdf
        # but it does not work on 16.04, so we do png instead
        # https://bugs.launchpad.net/ubuntu/+source/imagemagick/+bug/1796563
        viz_file_hq = os.path.join(
            viz_folder_hq,
            "image{:02d}_yes.png".format(i)
            if i in registed_index
            else "image{:02d}_no.png".format(i),
        )
        viz_file_lq = os.path.join(
            viz_folder_lq,
            "image{:02d}_yes.jpg".format(i)
            if i in registed_index
            else "image{:02d}_no.png".format(i),
        )
        plt.savefig(viz_file_hq, bbox_inches="tight")

        # Convert with imagemagick
        os.system(
            'convert -quality 75 -resize "640>" {} {}'.format(viz_file_hq, viz_file_lq)
        )

        plt.close()

    print(
        f"{len(img_paths)} images in bag, index: {np.array(registed_index)} registrated"
    )
    for i, mask in enumerate(registed_points_mask):
        print(f"\nindex :{registed_index[i]}, {mask.sum()}/{len(mask)} |")

    print("Done [{:.02f} s.]".format(time() - t_start))


def eval_colmap_results(dataset, args):
    """
    Computes the error using quaternions and translation vector for COLMAP
    """
    base_path = args.save_dir
    assert osp.exists(base_path)

    # Load visiblity and images
    image_path_list = dataset.img_paths
    f_names_list = dataset.f_names
    calib_list = dataset.calib_paths
    pair_ids = dataset.pair_ids

    # Load camera information
    calib_dict = load_calib(calib_list)

    # Check if colmap results exist. Otherwise, this whole bag is a fail.
    colmap_output_path = osp.join(base_path, "colmap_output_path")
    is_colmap_valid = os.path.exists(os.path.join(colmap_output_path, "0"))

    if is_colmap_valid:

        # Find the best colmap reconstruction
        # best_index = get_best_colmap_index(cfg)
        best_index = args.eval_best_index

        print("Computing pose errors")

        """
        num_cores = int(multiprocessing.cpu_count() * 0.9)
        # num_cores = int(len(os.sched_getaffinity(0)) * 0.9)
        result = Parallel(n_jobs=num_cores)(
            delayed(compute_stereo_metrics_from_colmap)(
                image_path_list[pair[0]],
                image_path_list[pair[1]],
                calib_dict[f_names_list[pair[0]]],
                calib_dict[f_names_list[pair[1]]],
                best_index,
                colmap_output_path,
            )
            for pair in tqdm(pair_ids)
        )
        """

        result = [
            compute_stereo_metrics_from_colmap(
                image_path_list[pair[0]],
                image_path_list[pair[1]],
                calib_dict[f_names_list[pair[0]]],
                calib_dict[f_names_list[pair[1]]],
                best_index,
                colmap_output_path,
            )
            for pair in tqdm(pair_ids)
        ]

    # Collect err_q, err_t from results
    err_dict = {}
    R_error, t_error = [], []

    for _i in range(len(pair_ids)):
        pair = pair_ids[_i]
        if is_colmap_valid:
            err_q = result[_i][0]
            err_t = result[_i][1]
        else:
            err_q = np.inf
            err_t = np.inf
        err_dict[f_names_list[pair[0]] + "-" + f_names_list[pair[1]]] = [
            err_q,
            err_t,
        ]
        R_error.append(err_q)
        t_error.append(err_t)

    # Finally, save packed errors
    save_h5(err_dict, osp.join(base_path, "colmap_pose_error.h5"))

    # pose auc
    angular_thresholds = [1, 2, 3, 4, 5, 10, 20]
    pose_errors = np.max(np.stack([R_error, t_error]), axis=0)
    aucs = pose_auc(pose_errors, angular_thresholds, True)

    with open(osp.join(base_path, "colmap_pose_error_auc.txt"), "w") as f:
        for key in aucs.keys():
            f.write(key + ":\n")
            f.write(f"{aucs[key]}\n")
            f.write("--------------\n")
    print(aucs)


def main():
    args = parse_args()

    if args.slurm:
        ray.init(address=os.environ["ip_head"])
    else:
        ray.init(
            num_cpus=math.ceil(args.n_workers * args.n_cpus_per_worker),
            num_gpus=math.ceil(args.n_workers * args.n_gpus_per_worker),
            local_mode=args.local_mode,
        )

    # Build dataset
    logger.info('Construct Dataset begin....')
    if args.dataset_name == "IMC":
        dataset = IMCValPairDataset(
            args.data_root,
            args.img_resize_max,
            args.n_imgs,
            subset_name=args.subset_name,
        )
    else:

        raise NotImplementedError

    # Matching:
    logger.info('Exhaustive matching begin....')
    if (
        osp.exists(osp.join(args.save_dir, "keypoints.h5"))
        and osp.exists(osp.join(args.save_dir, "matches.h5"))
        and not args.match_debug
    ):
        logger.info(
            "Keypoints and matches exists! Don't need to run exhaustive matching!"
        )
        keypoints_dict = load_h5(osp.join(args.save_dir, "keypoints.h5"))
        matches_dict = load_h5(osp.join(args.save_dir, "matches.h5"))
    else:
        logger.info("Exhaustive matching running!")
        keypoints_dict, matches_dict = exhaustive_matching(args, dataset)

    # Run colmap:
    logger.info("COLMAP running!")
    colmapRunner(dataset, keypoints_dict, matches_dict, args)

    # Visualzie colmap:
    if args.visual_enable:
        logger.info("Visualize COLMAP running!")
        visualize_colmap(dataset, keypoints_dict, args)

    # Eval colmap results:
    logger.info("Eval colmap results running!")
    # eval_colmap_results(dataset, args)

    pass


if __name__ == "__main__":
    main()
