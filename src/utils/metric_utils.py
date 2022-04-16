import numpy as np
import os
import cv2
import torch
import os.path as osp
from loguru import logger
from tools.data_prepare.sample_points_on_cad import sample_points_on_cad, model_diameter_from_bbox
from .colmap.read_write_model import qvec2rotmat, read_images_binary
from .colmap.eval_helper import quaternion_from_matrix


def convert_pose2T(pose):
    # pose: [R: 3*3, t: 3]
    R, t = pose
    return np.concatenate(
        [np.concatenate([R, t[:, None]], axis=1), [[0, 0, 0, 1]]], axis=0
    )  # 4*4


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt) ** 2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython

        IPython.embed()

    return err_q, err_t


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def compute_stereo_metrics_from_colmap(
    img1,
    img2,
    calib1,
    calib2,
    best_index,
    colmap_output_path,
    use_imc_pose_error_method="False",
):
    """Computes (pairwise) error metrics from Colmap results."""

    # Load COLMAP dR and dt

    # First read images.bin for the best reconstruction
    images_bin = read_images_binary(
        os.path.join(colmap_output_path, str(best_index), "images.bin")
    )

    # For each key check if images_bin[key].name = image_name
    R_1_actual, t_1_actual = None, None
    R_2_actual, t_2_actual = None, None
    for key in images_bin.keys():
        if images_bin[key].name == os.path.basename(img1):
            R_1_actual = qvec2rotmat(images_bin[key].qvec)
            t_1_actual = images_bin[key].tvec
        if images_bin[key].name == os.path.basename(img2):
            R_2_actual = qvec2rotmat(images_bin[key].qvec)
            t_2_actual = images_bin[key].tvec

    # Compute err_q and err_t only when R, t are not None
    err_q, err_t = np.inf, np.inf
    if (
        (R_1_actual is not None)
        and (R_2_actual is not None)
        and (t_1_actual is not None)
        and (t_2_actual is not None)
    ):
        # Compute dR, dt (actual)
        dR_act = np.dot(R_2_actual, R_1_actual.T)
        dt_act = t_2_actual - np.dot(dR_act, t_1_actual)

        # Get R, t from calibration information
        R_1, t_1 = calib1["R"], calib1["T"].reshape((3, 1))
        R_2, t_2 = calib2["R"], calib2["T"].reshape((3, 1))

        # Compute ground truth dR, dt
        dR = np.dot(R_2, R_1.T)
        dt = t_2 - np.dot(dR, t_1)  # (3,)

        # Save err_, err_t
        if use_imc_pose_error_method:
            err_q, err_t = evaluate_R_t(dR, dt, dR_act, dt_act)  # rad!
        else:
            err_q = angle_error_mat(dR_act, dR)  # err_R actually
            dt = dt.flatten()
            dt_act = dt_act.flatten()
            err_t = angle_error_vec(dt_act, dt)  # degree!
    return err_q, err_t


def pose_auc(errors, thresholds, ret_dict=False):
    if len(errors) == 0:
        aucs = [0 for i in thresholds]
    else:
        sort_idx = np.argsort(errors)
        errors = np.array(errors.copy())[sort_idx]
        recall = (np.arange(len(errors)) + 1) / len(errors)
        errors = np.r_[0.0, errors]
        recall = np.r_[0.0, recall]
        aucs = []
        for t in thresholds:
            last_index = np.searchsorted(errors, t)
            r = np.r_[recall[:last_index], recall[last_index - 1]]
            e = np.r_[errors[:last_index], t]
            aucs.append(np.trapz(r, x=e) / t)
    if ret_dict:
        return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}
    else:
        return aucs

def projection_2d_error(model_3D_pts, pose_pred, pose_targets, K):
    def project(xyz, K, RT):
        """
        xyz: [N, 3]
        K: [3, 3]
        RT: [3, 4]
        """
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy

    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_targets.shape[0] == 4:
        pose_targets = pose_targets[:3]

    model_2d_pred = project(model_3D_pts, K, pose_pred) # pose_pred: 3*4
    model_2d_targets = project(model_3D_pts, K, pose_targets)
    proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
    return proj_mean_diff

def add_metric(model_3D_pts, diameter, pose_pred, pose_target, percentage=0.1):
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_target.shape[0] == 4:
        pose_target = pose_target[:3]

    diameter = diameter * percentage
    model_pred = np.dot(model_3D_pts, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_target = np.dot(model_3D_pts, pose_target[:, :3].T) + pose_target[:, 3]

    mean_dist = np.mean(np.linalg.norm(model_pred - model_target, axis=-1))
    return mean_dist < diameter


# Evaluate query pose errors
def query_pose_error(pose_pred, pose_gt, unit='m'):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]

    # Convert results' unit to cm
    if unit == 'm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    elif unit == 'cm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
    elif unit == 'mm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) / 10
    else:
        raise NotImplementedError

    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance


def ransac_PnP(
    K,
    pts_2d,
    pts_3d,
    scale=1,
    pnp_reprojection_error=5,
    img_hw=None,
    use_pycolmap_ransac=False,
):
    """ solve pnp """
    if use_pycolmap_ransac:
        import pycolmap

        assert img_hw is not None and len(img_hw) == 2

        pts_2d = list(np.ascontiguousarray(pts_2d.astype(np.float64))[..., None]) # List(2*1)
        pts_3d = list(np.ascontiguousarray(pts_3d.astype(np.float64))[..., None]) # List(3*1)
        # pts_2d = pts_2d.astype(np.float64)
        # pts_3d = pts_3d.astype(np.float64)
        K = K.astype(np.float64)
        # Colmap pnp
        focal_length = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]
        cfg = {
            "model": "SIMPLE_PINHOLE",
            "width": int(img_hw[1]),
            "height": int(img_hw[0]),
            "params": [focal_length, cx, cy],
        }

        ret = pycolmap.absolute_pose_estimation(
            pts_2d, pts_3d, cfg, max_error_px=float(pnp_reprojection_error)
        )
        qvec = ret["qvec"]
        tvec = ret["tvec"]
        pose_homo = convert_pose2T([qvec2rotmat(qvec), tvec])
        # Make inliers:
        inliers = ret['inliers']
        if len(inliers) == 0:
            inliers = np.array([]).astype(np.bool)
        else:
            index = np.arange(0, len(pts_3d))
            inliers = index[inliers]

        return pose_homo[:3], pose_homo, inliers, True
    else:
        dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

        pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
        pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
        K = K.astype(np.float64)

        pts_3d *= scale
        state = None
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                K,
                dist_coeffs,
                reprojectionError=pnp_reprojection_error,
                iterationsCount=10000,
                flags=cv2.SOLVEPNP_EPNP,
            )
            # _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, dist_coeffs)

            rotation = cv2.Rodrigues(rvec)[0]

            tvec /= scale
            pose = np.concatenate([rotation, tvec], axis=-1)
            pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

            if inliers is None:
                inliers = np.array([]).astype(np.bool)
            state = True

            return pose, pose_homo, inliers, state
        except cv2.error:
            # print("CV ERROR")
            state = False
            return np.eye(4)[:3], np.eye(4), np.array([]).astype(np.bool), state


@torch.no_grad()
def compute_query_pose_errors(
    data, configs, compute_gt_proj_pose_error=True, training=False
):
    """
    Update:
        data(dict):{
            "R_errs": []
            "t_errs": []
            "inliers": []
        }
    """
    device = data["m_bids"].device
    model_unit = configs['model_unit'] if 'model_unit' in configs else 'm'

    m_bids = data["m_bids"].cpu().numpy()
    mkpts_3d = data["mkpts_3d_db"].cpu().numpy()
    mkpts_query = data["mkpts_query_f"].cpu().numpy()
    mkpts_query_c = data["mkpts_query_c"].cpu().numpy()
    img_orig_size = (
        torch.tensor(data["q_hw_i"]).numpy() * data["query_image_scale"].cpu().numpy()
    )  # B*2
    query_K = data["query_intrinsic"].cpu().numpy()
    query_pose_gt = data["query_pose_gt"].cpu().numpy()  # B*4*4
    # homo_warp = data['homo_warp'].cpu().numpy() if 'homo_warp' in data else None # B*3*3

    data.update({"R_errs": [], "t_errs": [], "inliers": []})
    data.update({"R_errs_c": [], "t_errs_c": [], "inliers_c": []})

    # Prepare query model for eval ADD metric
    if 'eval_ADD_metric' in configs:
        if configs['eval_ADD_metric'] and not training:
            image_path = data['query_image_path']
            model_path = osp.join(image_path.rsplit('/', 3)[0], 'model_eval.ply')
            if not osp.exists(model_path):
                logger.warning('Model_eval.ply not exists! try to use model.ply instead!')
                model_path = osp.join(image_path.rsplit('/', 3)[0], 'model.ply')
            diameter_file_path = osp.join(image_path.rsplit('/', 3)[0], 'diameter.txt')
            if not osp.exists(model_path):
                logger.warning(f'want to eval add metric, however model_eval.ply path:{model_path} not exists!')
            else:
                # Load model:
                model_vertices, bbox = sample_points_on_cad(model_path, 10000) # N*3
                # Load diameter:
                if osp.exists(diameter_file_path):
                    diameter = np.loadtxt(diameter_file_path)
                else:
                    diameter = model_diameter_from_bbox(bbox)
                    logger.warning(f'Diameter file not exists! Diameter compute from CAD model is {diameter}')
                
                data.update({"ADD":[]})

    pose_pred = []
    for bs in range(query_K.shape[0]):
        mask = m_bids == bs

        # FIXME: bug exists here!
        mkpts_query_f = mkpts_query[mask]
        if compute_gt_proj_pose_error:
            # Reproj mkpts:
            R = query_pose_gt[bs][:3, :3]  # 3*3
            t = query_pose_gt[bs][:3, [3]]  # 3*1
            mkpts_3d_cam = R @ mkpts_3d[mask].T + t  # 3*N
            mkpts_proj = (query_K[bs] @ mkpts_3d_cam).T  # N*3
            mkpts_query_gt = mkpts_proj[:, :2] / mkpts_proj[:, [2]]

            diff = np.linalg.norm(mkpts_query_f - mkpts_query_gt, axis=-1)
            mkpts_query_gt[diff > 6] = mkpts_query_f[
                diff > 6
            ]  # Use real pred to test real diff

        query_pose_pred, query_pose_pred_homo, inliers, state = ransac_PnP(
            query_K[bs],
            mkpts_query_f,
            mkpts_3d[mask],
            scale=configs["point_cloud_rescale"],
            img_hw=img_orig_size[bs].tolist(),
            pnp_reprojection_error=configs["pnp_reprojection_error"],
            use_pycolmap_ransac=configs["use_pycolmap_ransac"],
        )
        pose_pred.append(query_pose_pred_homo)

        query_pose_pred_c, query_pose_pred_homo_c, inliers_c, state_c = ransac_PnP(
            query_K[bs],
            mkpts_query_c[mask],
            mkpts_3d[mask],
            scale=configs["point_cloud_rescale"],
            img_hw=img_orig_size[bs].tolist(),
            pnp_reprojection_error=configs["pnp_reprojection_error"],
            use_pycolmap_ransac=configs["use_pycolmap_ransac"],
        )

        if configs["enable_post_optimization"] and state is not False and not training:
            from src.NeuralSfM.loc.optimizer import Optimizer

            optimizer = Optimizer(configs["post_optimization_configs"])
            scale = data["query_image_scale"][bs].cpu().numpy()[None]  # [1*2]
            inliers_mask = np.full(
                (mkpts_3d[mask].shape[0],), 0, dtype=np.bool
            )  # All False
            if configs["post_optimization_configs"]["use_fine_pose_as_init"]:
                initial_pose = query_pose_pred  # [3*4]
                inliers_mask[inliers] = True
            else:
                initial_pose = query_pose_pred_c  # [3*4]
                inliers_mask[inliers_c] = True
            mkpts_3d_inlier = mkpts_3d[mask][inliers_mask]
            mkpts_2d_c_inlier = mkpts_query_c[mask][inliers_mask]
            mkpts_2d_f_inlier = mkpts_query_f[mask][inliers_mask]

            feature_3d = data["desc3d_db_selected"][mask][inliers_mask]
            feature_2d_window = data["query_feat_f_unfolded"][mask][inliers_mask]

            query_pose_pred_refined = optimizer.start_optimize(
                [initial_pose[:, :3], initial_pose[:, 3]],
                query_K[bs],
                mkpts3d=mkpts_3d_inlier,
                mkpts2d_c=mkpts_2d_c_inlier,
                mkpts2d_f=mkpts_2d_f_inlier,
                scale=scale,
                feature_3d=feature_3d,
                feature_2d_window=feature_2d_window,
                feature_distance_map_temperature=configs["post_optimization_configs"][
                    "feature_distance_map_temperature"
                ],
                device=device,
                point_cloud_scale=configs["point_cloud_rescale"],
            )
            query_pose_pred = query_pose_pred_refined

            # For debug:
            R_err_before, t_err_before = query_pose_error(
                initial_pose, query_pose_gt[bs], unit=model_unit
            )
            R_err_after, t_err_after = query_pose_error(
                query_pose_pred_refined, query_pose_gt[bs], unit=model_unit
            )
            R_err_decrease = R_err_before - R_err_after
            t_err_decrease = t_err_before - t_err_after

            if R_err_decrease < 0 or t_err_decrease < 0:
                R_err_decrease = R_err_decrease

        if query_pose_pred is None:
            data["R_errs"].append(np.inf)
            data["t_errs"].append(np.inf)
            data["inliers"].append(np.array([]).astype(np.bool))
            if "ADD" in data:
                data['ADD'].append(False)
        else:
            R_err, t_err = query_pose_error(query_pose_pred, query_pose_gt[bs], unit=model_unit)
            data["R_errs"].append(R_err)
            data["t_errs"].append(t_err)
            data["inliers"].append(inliers)

            if "ADD" in data:
                add_result = add_metric(model_vertices, diameter, pose_pred=query_pose_pred, pose_target=query_pose_gt[bs])
                data["ADD"].append(add_result)

        if query_pose_pred_c is None:
            data["R_errs_c"].append(np.inf)
            data["t_errs_c"].append(np.inf)
            data["inliers_c"].append(np.array([]).astype(np.bool))
        else:
            R_err, t_err = query_pose_error(query_pose_pred_c, query_pose_gt[bs], unit=model_unit)
            data["R_errs_c"].append(R_err)
            data["t_errs_c"].append(t_err)
            data["inliers_c"].append(inliers_c)

        if compute_gt_proj_pose_error:
            data.update({"R_errs_gt": [], "t_errs_gt": [], "inliers_gt": []})
            query_pose_pred_gt, query_pose_pred_homo_gt, inliers_gt, state = ransac_PnP(
                query_K[bs],
                mkpts_query_gt,  # FIXME: change to mkpts_query
                mkpts_3d[mask],
                scale=configs["point_cloud_rescale"],
                pnp_reprojection_error=configs["pnp_reprojection_error"],
            )

            if query_pose_pred_gt is None:
                data["R_errs_gt"].append(np.inf)
                data["t_errs_gt"].append(np.inf)
                data["inliers_gt"].append(np.array([]).astype(np.bool))
            else:
                R_err, t_err = query_pose_error(query_pose_pred_gt, query_pose_gt[bs],unit=model_unit)
                data["R_errs_gt"].append(R_err)
                data["t_errs_gt"].append(t_err)
                data["inliers_gt"].append(inliers_gt)

    pose_pred = np.stack(pose_pred)  # [B*4*4]
    data.update({"pose_pred": pose_pred})


def aggregate_metrics(metrics, pose_thres=[1, 3, 5], proj2d_thres=5):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4
    """
    # filter duplicates
    # unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    # unq_ids = list(unq_ids.values())
    # logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    R_errs = metrics["R_errs"]
    t_errs = metrics["t_errs"]

    agg_metric = {}
    for pose_threshold in pose_thres:
        agg_metric[f"{pose_threshold}cm@{pose_threshold}degree"] = np.mean(
            (np.array(R_errs) < pose_threshold) & (np.array(t_errs) < pose_threshold)
        )

    if "R_errs_coarse" in metrics:
        R_errs_coarse = metrics["R_errs_coarse"]
        t_errs_coarse = metrics["t_errs_coarse"]

        for threshold in pose_thres:
            agg_metric[f"{threshold}cm@{threshold}degree coarse"] = np.mean(
                (np.array(R_errs_coarse) < threshold)
                & (np.array(t_errs_coarse) < threshold)
            )
    
    if "ADD_metric" in metrics:
        ADD_metric = metrics['ADD_metric']
        agg_metric["ADD metric"] = np.mean(ADD_metric)

    return agg_metric
