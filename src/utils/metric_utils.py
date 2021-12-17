import numpy as np
import os
from .colmap.read_write_model import qvec2rotmat, read_images_binary
from .colmap.eval_helper import quaternion_from_matrix

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
            err_q, err_t = evaluate_R_t(dR, dt, dR_act, dt_act) # rad!
        else:
            err_q = angle_error_mat(dR_act, dR)  # err_R actually
            dt = dt.flatten()
            dt_act = dt_act.flatten()
            err_t = angle_error_vec(dt_act, dt) # degree!
    return err_q, err_t

def pose_auc(errors, thresholds, ret_dict=False):
    if len(errors) == 0:
        aucs = [0 for i in thresholds]
    else:
        sort_idx = np.argsort(errors)
        errors = np.array(errors.copy())[sort_idx]
        recall = (np.arange(len(errors)) + 1) / len(errors)
        errors = np.r_[0., errors]
        recall = np.r_[0., recall]
        aucs = []
        for t in thresholds:
            last_index = np.searchsorted(errors, t)
            r = np.r_[recall[:last_index], recall[last_index-1]]
            e = np.r_[errors[:last_index], t]
            aucs.append(np.trapz(r, x=e)/t)
    if ret_dict:
        return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}
    else:
        return aucs