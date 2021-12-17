import argparse
from typing import ChainMap
import ray
import os
import os.path as osp

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
import math
import pytorch_lightning as pl
import torch
import numpy as np
from ray.actor import ActorHandle
from loguru import logger

from src.datasets.loftr_coarse_dataset import loftr_coarse_dataset
from src.utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict
from src.utils.misc import lower_config
from src.utils.torch_utils import update_state_dict, STATE_DICT_MAPPER
from src.utils.data_io import save_h5, load_h5

from ..loftr_config.default import get_cfg_defaults
from ..loftr_for_sfm.loftr import Matcher_LoFTR
from ..extractors import build_extractor
from ..loftr_for_sfm.utils.detector_wrapper import DetectorWrapper, DetectorWrapperTwoView
from .coarse_matcher_utils import agg_groupby_2d, extract_geo_model_inliers, Match2Kpts

cfgs = {
    'data':{
        'img_resize': 512,
        'df': 8,
        'n_imgs': None, # For debug
        'shuffle': True
    },
    'matcher':{
        'model':{
            'cfg_path': '',
            'weight_path': '',
            'seed': 666,
        },
        'inlier_only': False,
        'ransac':{
            'geo_model': 'F',
            'ransac_method': 'DEGENSAC',
            'pixel_thr': 1.0,
            'max_iters': 10000,
            'conf_thr': 0.99999
        },
    },

    'cache': True,
    'ray':{
        'slurm': False,
        'n_workers': 1,
        'n_cpus_per_worker': 1,
        'n_gpus_per_worker': 1,
        'local_mode': False,
    }
}

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Match and refinement model related parameters

    # Date related
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
    cfg.merge_from_file(args['cfg_path'])
    pl.seed_everything(args['seed'])

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
    matcher = Matcher_LoFTR(config=match_cfg)
    # load checkpoints
    state_dict = torch.load(args['weight_path'], map_location="cpu")["state_dict"]
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

    detector.eval()
    matcher.eval()

    return detector, matcher


def extract_preds(data):
    """extract predictions assuming bs==1"""
    m_bids = data["m_bids"].cpu().numpy()
    assert (np.unique(m_bids) == 0).all()
    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()
    mconfs = data["mconf"].cpu().numpy()

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
    mkpts0, mkpts1, mconfs, detector_kpts_mask = extract_preds(data)
    K0 = data["K0"][0].cpu().numpy() if args['geo_model'] == "E" else None
    K1 = data["K1"][0].cpu().numpy() if args['geo_model'] == "E" else None
    if len(mkpts0) >= 8:
        inliers = extract_geo_model_inliers(
            mkpts0,
            mkpts1,
            mconfs,
            args['geo_model'],
            args['ransac_method'],
            args['pixel_thr'],
            args['max_iters'],
            args['conf_thr'],
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
def extract_matches(data, detector=None, matcher=None, ransac_args=None, inlier_only=True):
    # 1. inference
    detector(data)
    matcher(data)

    # 2. run RANSAC and extract inliers
    mkpts0, mkpts1, mconfs, detector_kpts_mask = (
        extract_inliers(data, ransac_args) if inlier_only else extract_preds(data)
    )
    del data
    torch.cuda.empty_cache()
    return (mkpts0, mkpts1, mconfs, detector_kpts_mask)


@torch.no_grad()
def match_worker(dataset, subset_ids, args, pba=None):
    """extract matches from part of the possible image pair permutations"""
    detector, matcher = build_model(args['model'])
    detector.cuda()
    matcher.cuda()
    matches = {}
    # match all permutations
    for subset_id in subset_ids:
        data = dataset[subset_id]
        f_name0, f_name1 = data['pair_key']
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        mkpts0, mkpts1, mconfs, detector_kpts_mask = extract_matches(
            data_c,
            detector=detector,
            matcher=matcher,
            args=args['ransac'],
            inlier_only=args['inlier_only'],
        )

        # Extract matches (kpts-pairs & scores)
        matches["-".join([f_name0, f_name1])] = np.concatenate(
            [mkpts0, mkpts1, mconfs[:, None]], -1
        )  # (N, 5)

        if pba is not None:
            pba.update.remote(1)
    return matches

@ray.remote(num_cpus=1, num_gpus=1, max_calls=1)  # release gpu after finishing
def match_worker_ray_wrapper(dataset, subset_ids, args, pba: ActorHandle):
    match_worker(dataset, subset_ids, args, pba)

def keypoint_worker(name_kpts, pba=None):
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
def keypoints_worker_ray_wrapper(name_kpts, pba: ActorHandle):
    keypoint_worker(name_kpts, pba)


def update_matches(matches, keypoints, pba=None):
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
        if pba is not None:
            pba.update.remote(1)

    return ret_matches

@ray.remote(num_cpus=1)
def update_matches_ray_wrapper(matches, keypoints, pba: ActorHandle):
    update_matches(matches, keypoints, pba)


def transform_keypoints(keypoints, pba=None):
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
        if pba is not None:
            pba.update.remote(1)
    return ret_kpts, ret_scores

@ray.remote(num_cpus=1)
def transform_keypoints_ray_wrapper(keypoints, pba: ActorHandle):
    transform_keypoints(keypoints, pba)

def loftr_coarse_matching_ray(image_lists, covis_pairs_out, feature_out, matches_out):
    # Initial ray:
    cfg_ray = cfgs['ray']
    if cfg_ray['slurm']:
        ray.init(address=os.environ["ip_head"])
    else:
        ray.init(
            num_cpus=math.ceil(cfg_ray['n_workers'] * cfg_ray['n_cpus_per_worker']),
            num_gpus=math.ceil(cfg_ray['n_workers'] * cfg_ray['n_gpus_per_worker']),
            local_mode=cfg_ray['local_mode'],
        )
    
    # Build dataset:
    dataset = loftr_coarse_dataset(cfgs['data'], image_lists, covis_pairs_out)

    # Construct directory
    base_dir = feature_out.rsplit('/', 1)[0]
    os.makedirs(base_dir, exist_ok=True)

    # Matcher runner
    cache_dir = osp.join(feature_out.rsplit('/', 1)[0], 'raw_matches.h5')
    if cfgs['cache'] and osp.exists(cache_dir):
        matches = load_h5(
            cache_dir, transform_slash=True
        )
        logger.info("Caches raw matches loaded!")
    else:
        pb = ProgressBar(len(dataset), "Matching image pairs...")
        all_subset_ids = chunk_index(
            len(dataset), math.ceil(len(dataset) / cfgs['n_workers'])
        )
        obj_refs = [
            match_worker_ray_wrapper.remote(dataset, subset_ids, cfgs['matcher'], pb.actor)
            for subset_ids in all_subset_ids
        ]
        pb.print_until_done()
        results = ray.get(obj_refs)
        matches = dict(ChainMap(*results))
        logger.info("Matcher finish!")

        # over write anyway
        save_h5(matches, cache_dir)
        logger.info(f"Raw matches cached: {cache_dir}")

    # Combine keypoints
    n_imgs = len(dataset.f_names)
    pb = ProgressBar(n_imgs, "Combine keypoints")
    all_kpts = Match2Kpts(matches, dataset.f_names)
    sub_kpts = chunks(all_kpts, math.ceil(n_imgs / cfgs['n_workers']))
    obj_refs = [keypoints_worker_ray_wrapper.remote(sub_kpt, pb.actor) for sub_kpt in sub_kpts]
    pb.print_until_done()
    keypoints = dict(ChainMap(*ray.get(obj_refs)))
    logger.info("Combine keypoints finish!")

    # Convert keypoints match to keypoints indexs
    pb = ProgressBar(len(matches), "Updating matches...")
    _keypoints_ref = ray.put(keypoints)
    obj_refs = [
        update_matches_ray_wrapper.remote(sub_matches, _keypoints_ref, pb.actor)
        for sub_matches in split_dict(matches, math.ceil(len(matches) / cfg_ray['n_workers']))
    ]
    pb.print_until_done()
    updated_matches = dict(ChainMap(*ray.get(obj_refs)))

    # Post process keypoints:
    keypoints = {
        k: v for k, v in keypoints.items() if isinstance(v, dict)
    }  # assume filename in f'{xxx}_{yyy}' format
    pb = ProgressBar(len(keypoints), "Post-processing keypoints...")
    obj_refs = [
        transform_keypoints_ray_wrapper.remote(sub_kpts, pb.actor)
        for sub_kpts in split_dict(
            keypoints, math.ceil(len(keypoints) / cfg_ray['n_workers'])
        )
    ]
    pb.print_until_done()
    kpts_scores = ray.get(obj_refs)
    final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
    final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    # Save finial results:
    save_h5(final_keypoints, feature_out)
    save_h5(updated_matches, matches_out)
    # TODO: change save feature and keypoints to onepose format

    return final_keypoints, updated_matches

def loftr_coarse_matching(image_lists, covis_pairs_out, feature_out, matches_out):
    # Build dataset:
    dataset = loftr_coarse_dataset(cfgs['data'], image_lists, covis_pairs_out)

    # Construct directory
    base_dir = feature_out.rsplit('/', 1)[0]
    os.makedirs(base_dir, exist_ok=True)

    # Matcher runner
    cache_dir = osp.join(feature_out.rsplit('/', 1)[0], 'raw_matches.h5')
    if cfgs['cache'] and osp.exists(cache_dir):
        matches = load_h5(
            cache_dir, transform_slash=True
        )
        logger.info("Caches raw matches loaded!")
    else:
        all_subset_ids = chunk_index(
            len(dataset), math.ceil(len(dataset) / cfgs['n_workers'])
        )
        obj_refs = [
            match_worker_ray_wrapper.remote(dataset, subset_ids, cfgs['matcher'], pb.actor)
            for subset_ids in all_subset_ids
        ]
        results = ray.get(obj_refs)
        matches = dict(ChainMap(*results))
        logger.info("Matcher finish!")

        # over write anyway
        save_h5(matches, cache_dir)
        logger.info(f"Raw matches cached: {cache_dir}")

    # Combine keypoints
    n_imgs = len(dataset.f_names)
    pb = ProgressBar(n_imgs, "Combine keypoints")
    all_kpts = Match2Kpts(matches, dataset.f_names)
    sub_kpts = chunks(all_kpts, math.ceil(n_imgs / cfgs['n_workers']))
    obj_refs = [keypoints_worker_ray_wrapper.remote(sub_kpt, pb.actor) for sub_kpt in sub_kpts]
    pb.print_until_done()
    keypoints = dict(ChainMap(*ray.get(obj_refs)))
    logger.info("Combine keypoints finish!")

    # Convert keypoints match to keypoints indexs
    pb = ProgressBar(len(matches), "Updating matches...")
    _keypoints_ref = ray.put(keypoints)
    obj_refs = [
        update_matches_ray_wrapper.remote(sub_matches, _keypoints_ref, pb.actor)
        for sub_matches in split_dict(matches, math.ceil(len(matches) / cfg_ray['n_workers']))
    ]
    pb.print_until_done()
    updated_matches = dict(ChainMap(*ray.get(obj_refs)))

    # Post process keypoints:
    keypoints = {
        k: v for k, v in keypoints.items() if isinstance(v, dict)
    }  # assume filename in f'{xxx}_{yyy}' format
    pb = ProgressBar(len(keypoints), "Post-processing keypoints...")
    obj_refs = [
        transform_keypoints_ray_wrapper.remote(sub_kpts, pb.actor)
        for sub_kpts in split_dict(
            keypoints, math.ceil(len(keypoints) / cfg_ray['n_workers'])
        )
    ]
    pb.print_until_done()
    kpts_scores = ray.get(obj_refs)
    final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
    final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    # Save finial results:
    save_h5(final_keypoints, feature_out)
    save_h5(updated_matches, matches_out)
    # TODO: change save feature and keypoints to onepose format

    return final_keypoints, updated_matches