from typing import List
import torch
import numpy as np
from src.utils.utils import estimate_pose, estimate_pose_degensac, estimate_pose_magsac


def two_view_pose_initialize(data, config):
    """ 
    Update:
        data (dict):{
            "initial_pose" List[List[torch.tensor] : 3 (R 3*3,t 3*1,inlier_mask N*1)] : N
        }
    """
    method = config["pose_estimation_method"]  # RANSAC
    pixel_thr = config["ransac_pixel_thr"]  # 1.0
    conf = config["ransac_conf"]  # 0.99999
    max_iters = config["ransac_max_iters"]  # 1000

    data.update({"initial_pose": []})
    device = data["m_bids"].device

    m_bids = data["m_bids"].cpu().numpy()
    pts0 = data["mkpts0_f"].cpu().numpy()
    pts1 = data["mkpts1_f"].cpu().numpy()
    K0 = data["K0"].cpu().numpy()
    K1 = data["K1"].cpu().numpy()

    # TODO: parallel evaluation
    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        if method == "RANSAC":
            ret = estimate_pose(
                pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf
            )
        elif method == "DEGENSAC":
            ret = estimate_pose_degensac(
                pts0[mask],
                pts1[mask],
                K0[bs],
                K1[bs],
                pixel_thr,
                conf=conf,
                max_iters=max_iters,
            )
        elif method == "MAGSAC":
            ret = estimate_pose_magsac(
                pts0[mask],
                pts1[mask],
                K0[bs],
                K1[bs],
                config.TRAINER.USE_MAGSACPP,
                conf=conf,
                max_iters=max_iters,
            )
        else:
            raise NotImplementedError

        if ret is not None:
            # convert numpy.array to torch.tensor
            ret = list(map(lambda a: torch.from_numpy(a).to(device).float(), ret))
            ret[1] = ret[1][:, None]
            ret[2] = ret[2].bool() # N
            data["initial_pose"].append(ret)

