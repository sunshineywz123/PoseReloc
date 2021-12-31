import ray
import os

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
import pytorch_lightning as pl
import torch
import numpy as np
from ray.actor import ActorHandle
from tqdm import tqdm

from src.utils.misc import lower_config
from src.utils.torch_utils import update_state_dict, STATE_DICT_MAPPER

from .coarse_matcher_utils import agg_groupby_2d, extract_geo_model_inliers
from ..loftr_config.default import get_cfg_defaults
from ..loftr_for_sfm.loftr_sfm import LoFTR_SfM
from ..extractors import build_extractor
from ..loftr_for_sfm.utils.detector_wrapper import DetectorWrapper, DetectorWrapperTwoView


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))

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
    matcher = LoFTR_SfM(config=match_cfg)
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

    subset_ids = tqdm(subset_ids) if pba is None else subset_ids

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
            ransac_args=args['ransac'],
            inlier_only=args['inlier_only'],
        )

        # Extract matches (kpts-pairs & scores)
        matches[args['pair_name_split'].join([f_name0, f_name1])] = np.concatenate(
            [mkpts0, mkpts1, mconfs[:, None]], -1
        )  # (N, 5)

        if pba is not None:
            pba.update.remote(1)
    return matches

@ray.remote(num_cpus=1, num_gpus=0.25)  # release gpu after finishing
# @ray.remote(num_cpus=1, num_gpus=1)  # release gpu after finishing
def match_worker_ray_wrapper(dataset, subset_ids, args, pba: ActorHandle):
    return match_worker(dataset, subset_ids, args, pba)

def keypoint_worker(name_kpts, pba=None):
    """merge keypoints associated with one image.
    python >= 3.7 only.
    """
    keypoints = {}
    name_kpts = tqdm(name_kpts) if pba is None else name_kpts
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

        if pba is not None:
            pba.update.remote(1)
    return keypoints

@ray.remote(num_cpus=1)
def keypoints_worker_ray_wrapper(name_kpts, pba: ActorHandle):
    return keypoint_worker(name_kpts, pba)


def update_matches(matches, keypoints, pba=None, **kwargs):
    # convert match to indices
    ret_matches = {}

    matches_items = tqdm(matches.items()) if pba is None else matches.items()

    for k, v in matches_items:
        mkpts0, mkpts1 = (
            map(tuple, v[:, :2].astype(int)),
            map(tuple, v[:, 2:4].astype(int)),
        )
        name0, name1 = k.split(kwargs['pair_name_split'])
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

        ret_matches[k] = mids.astype(int)  # (N,2)
        if pba is not None:
            pba.update.remote(1)

    return ret_matches

@ray.remote(num_cpus=1)
def update_matches_ray_wrapper(matches, keypoints, pba: ActorHandle, **kwargs):
    return update_matches(matches, keypoints, pba, **kwargs)


def transform_keypoints(keypoints, pba=None):
    """assume keypoints sorted w.r.t. score"""
    ret_kpts = {}
    ret_scores = {}

    keypoints_items = tqdm(keypoints.items()) if pba is None else keypoints.items()
    for k, v in keypoints_items:
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
    return transform_keypoints(keypoints, pba)