import torch
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
from src.utils.utils import (
    estimate_pose,
    estimate_pose_degensac,
    estimate_pose_magsac,
    compute_pose_error,
    pose_auc,
    epidist_prec,
)


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0 ** 2 * (
        1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
        + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2)
    )  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data["T_0to1"][:, :3, 3])
    E_mat = Tx @ data["T_0to1"][:, :3, :3]

    m_bids = data["m_bids"]
    pts0 = data["mkpts0_f"]
    pts1 = data["mkpts1_f"]

    epi_errs = []
    # TODO: Parallel evaluation
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(
                pts0[mask], pts1[mask], E_mat[bs], data["K0"][bs], data["K1"][bs]
            )
        )
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({"epi_errs": epi_errs})


def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    method = config.TRAINER.POSE_ESTIMATION_METHOD  # RANSAC
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 1.0
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    max_iters = config.TRAINER.RANSAC_MAX_ITERS  # 1000
    data.update({"R_errs": [], "t_errs": [], "inliers": []})

    m_bids = data["m_bids"].cpu().numpy()
    pts0 = data["mkpts0_f"].cpu().numpy()
    pts1 = data["mkpts1_f"].cpu().numpy()
    K0 = data["K0"].cpu().numpy()
    K1 = data["K1"].cpu().numpy()
    T_0to1 = data["T_0to1"].cpu().numpy()

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

        if ret is None:
            data["R_errs"].append(np.inf)
            data["t_errs"].append(np.inf)
            data["inliers"].append(np.array([]).astype(np.bool))
        else:
            R, t, inliers = ret
            t_errs, R_errs = compute_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data["R_errs"].append(R_errs)
            data["t_errs"].append(t_errs)
            data["inliers"].append(inliers)


def compute_pose_errors_for_refined_pose(poses, data):
    """ 
    Parameters:
    ---------------
    poses : List[List[R : torch.tensor 3*3, t : torch.tensor 3*1]] [N]

    Return:
    ---------------
    errors : Dict{"R_errs":List[float] [N],
                  "t_errs":List[float] [N]}
    """
    T_0to1 = data["T_0to1"].cpu().numpy()
    errors = {"R_errs": [], "t_errs": []}

    # TODO: parallel evaluation
    for bs, pose in enumerate(poses):
        if len(pose) is None:
            errors["R_errs"].append(np.inf)
            errors["t_errs"].append(np.inf)
        else:
            R, t = map(lambda a: a.cpu().numpy(), pose)
            t_errs, R_errs = compute_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            errors["R_errs"].append(R_errs)
            errors["t_errs"].append(t_errs)
    return errors


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics["identifiers"]))
    unq_ids = list(unq_ids.values())
    logger.info(f"Aggregating metrics over {len(unq_ids)} unique items...")

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics["R_errs"], metrics["t_errs"]]), axis=0)[
        unq_ids
    ]
    aucs = pose_auc(pose_errors, angular_thresholds, True)  # (auc@5, auc@10, auc@20)

    pose_refined_aucs = {}
    if 'R_direct_refined_errs' in metrics:
        pose_errors = np.max(np.stack([metrics["R_direct_refined_errs"], metrics["t_direct_refined_errs"]]), axis=0)[
            unq_ids
        ]
        pose_refined_aucs['direct_refined_aucs'] = pose_auc(pose_errors, angular_thresholds, True)  # (auc@5, auc@10, auc@20)
    if 'R_feature_based_refined_errs' in metrics:
        pose_errors = np.max(np.stack([metrics["R_feature_based_refined_errs"], metrics["t_feature_based_refined_errs"]]), axis=0)[
            unq_ids
        ]
        pose_refined_aucs['feature_based_refined_aucs'] = pose_auc(pose_errors, angular_thresholds, True)  # (auc@5, auc@10, auc@20)


    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(
        np.array(metrics["epi_errs"], dtype=object)[unq_ids], dist_thresholds, True
    )  # (prec@epi_err_thr)

    return {**aucs, **precs, **pose_refined_aucs}

'''
def aggregate_metrics(metrics,epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4
    """
    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)
    aucs = pose_auc(pose_errors, angular_thresholds, True)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(metrics['epi_errs'], dist_thresholds, True)  # (prec@5e-04)

    return {**aucs, **precs}
'''
