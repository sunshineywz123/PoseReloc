import torch
import pytorch_lightning as pl
import numpy as np
from ray.actor import ActorHandle
from tqdm import tqdm
import os.path as osp

from src.NeuralSfM.loftr_config.default import get_cfg_defaults
from src.NeuralSfM.loftr_for_sfm import LoFTR_SfM
from src.utils.misc import lower_config
from src.utils.torch_utils import update_state_dict, STATE_DICT_MAPPER
from src.NeuralSfM.extractors import build_extractor
from src.NeuralSfM.loftr_for_sfm.utils.detector_wrapper import DetectorWrapper, DetectorWrapperTwoView
from .utils import sample_feature_from_unfold_featuremap
from ..visualization.draw_plots import draw_matches, draw_local_heatmaps


def build_model(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_path)
    pl.seed_everything(args.seed)

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
    # pose_depth_refiner = PoseDepthRefinement(lower_config(cfg.LOFTR_SFM))
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
    return detector, matcher


def extract_preds(data):
    """extract predictions assuming bs==1"""
    m_bids = data["m_bids"].cpu().numpy()
    assert (np.unique(m_bids) == 0).all()
    mkpts0_c = data["mkpts0_c"].cpu().numpy()
    mkpts1_c = data["mkpts1_c"].cpu().numpy()
    mkpts0_f = data["mkpts0_f"].cpu().numpy()
    mkpts1_f = data["mkpts1_f"].cpu().numpy()
    mkpts0_idx = data["mkpts0_idx"].cpu().numpy() # from original dataset
    scale0 = data["scale0"].cpu().numpy()
    scale1 = data["scale1"].cpu().numpy()
    # mconfs = data["mconf"].cpu().numpy()

    # Get feature response map
    feat_f0_unfold = data["feat_f0_unfold"]
    feat_f1_unfold = data["feat_f1_unfold"]
    query_features = sample_feature_from_unfold_featuremap(feat_f0_unfold)
    distance_map = torch.linalg.norm(
        query_features.unsqueeze(1) - feat_f1_unfold, dim=-1, keepdim=True
    )  # L*WW*1
    distance_map = distance_map.cpu().numpy()
    # TODO: use feature distance map or use feature directly may be different in latter optimization.

    return mkpts0_c, mkpts1_c, mkpts0_f, mkpts1_f, distance_map, mkpts0_idx, scale0, scale1


def extract_results(
    data,
    detector=None,
    matcher=None,
    refiner=None,
    refine_args={},
    extract_preds_args={},
    inlier_only=True,
):
    # 1. inference
    detector(data)
    matcher(data)
    refiner(data, **refine_args) if refiner is not None else None
    # 2. extract match and refined poses
    mkpts0_c, mkpts1_c, mkpts0_f, mkpts1_f, distance_map, mkpts0_idx, scale0, scale1 = extract_preds(
        data, **extract_preds_args
    )
    del data
    torch.cuda.empty_cache()
    return (mkpts0_c, mkpts1_c, mkpts0_f, mkpts1_f, distance_map, mkpts0_idx, scale0, scale1)


# Used for two view match and pose & depth refinement
# num_cpus=args.n_cpus_per_worker, num_gpus = args.n_gpus_per_worker
# @ray.remote(num_cpus=1, num_gpus=1, max_calls=1)  # release gpu after finishing
@torch.no_grad()
def matchWorker(dataset, subset_ids, detector, matcher, args, debug=False, pba: ActorHandle = None):
    """extract matches from part of the possible image pair permutations"""
    # detector, matcher = build_model(args)
    detector.cuda()
    matcher.cuda()
    results_dict = {}
    # match all permutations
    for subset_id in tqdm(subset_ids):
        data = dataset[subset_id]
        frameID0, frameID1 = data["frame0_colmap_id"], data["frame1_colmap_id"]
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }

        mkpts0_c, mkpts1_c, mkpts0_f, mkpts1_f, distance_map, mkpts0_idx, scale0, scale1 = extract_results(
            data_c, detector=detector, matcher=matcher,
        )

        # 3. extract results
        # matches['-'.join([frameID0, FrameID1])] = np.concatenate([mkpts0, mkpts1, mconfs[:, None], detector_kpts_mask[:,None], multiscale_mask[:,None]], -1)  # (N, 7)
        pair_name = "-".join([str(frameID0), str(frameID1)])
        results_dict[pair_name] = { # colmap frame id
            "mkpts0_c": mkpts0_c, # N*2
            "mkpts1_c": mkpts1_c,
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "mkpts0_idx": mkpts0_idx,
            "distance_map": distance_map, # N*WW*1
            "scale0": scale0, # 1*2
            "scale1": scale1
        }
        if pba is not None:
            pba.update.remote(1)

        if debug:
            # Output match and distance patch
            draw_matches(data, results_dict[pair_name], save_path=osp.join("./test_match_pair", pair_name+'.png'))
            draw_local_heatmaps(data, distance_map, mkpts1_c, save_path=osp.join("./test_local_heatmaps", pair_name+'.png'))

    return results_dict