from functools import partial

from src.neural_sfm.operator.utils import findOverlapIndex
import cv2
import numpy as np
import torch
import open3d as o3d
import os.path as osp
import os
from submodules.DeepLM import Solve
from .optimization_loss import (
    triangulation_reprojection_error,
    triangulation_mid_point_error,
)
from pytorch3d import transforms
from loguru import logger


@torch.no_grad()
def threeD_points_initialize_opencv(
    data, save_path=None, triangulate_inlier_only=True, debug=False
):
    """
    Use initialized pose to triangulate two-view 3D points to initial depth of image0 
    The world coordinate is defined to camera_0 camear coordinate
    Parameter:
    ----------
    data: dict
    save_path : str, optional
        whether save triangulated point cloud
    triangulate_inlier_only : bool, optional
        if true : only triangulate inlier keypoints, inliers are extractes in pose initializtion
        else : triangulate all keypoints
    debug : bool, optional
        if true : save intermediate results for dump and visualize

    Update: 
    ----------
    dict: {
        'threeD_init' List[numpy.array : N*3]: N_bs
        'depth0_init' List[numpy.array : N] : N_bs
        }
    """
    m_bids = data["m_bids"].cpu().numpy()
    pts0 = data["mkpts0_f"].cpu().numpy()
    pts1 = data["mkpts1_f"].cpu().numpy()
    K0 = data["K0"].cpu().numpy()
    K1 = data["K1"].cpu().numpy()
    device = data["m_bids"].device

    # TODO: parallel evaluation
    for bs in range(K0.shape[0]):
        mask = m_bids == bs

        if data["initial_pose"][bs] is not None:
            R, T, inlier_mask = map(lambda a: a.cpu().numpy(), data["initial_pose"][bs])
            mask = mask & inlier_mask if triangulate_inlier_only else mask
        else:
            threeD_coords, camera0_depth = None, None
            reprojection_coord0, reprojection_coord1 = None, None
            camera0_gt_point_cloud = None
        T0 = np.hstack([np.eye(3), np.zeros((3,))[:, None]])
        T1 = np.hstack([R, T])
        triangulated_points = cv2.triangulatePoints(
            np.dot(K0[bs], T0), np.dot(K1[bs], T1), pts0[mask].T, pts1[mask].T
        )
        threeD_coords = cv2.convertPointsFromHomogeneous(triangulated_points.T).squeeze(
            1
        )  # N*3
        camera0_depth = np.dot(K0[bs], threeD_coords.T)[-1].T

        # calculate camera0 reprojection error
        reprojection_error0, reprojection_coord0 = reprojection_error_in_camera_coord(
            K0[bs], threeD_coords, pts0[mask]
        )
        mean_reprojection_error0 = np.mean(reprojection_error0)

        # calculate camera1 reprojection error
        threeD_coords_in_camera1 = np.dot(R, threeD_coords.T) + T
        reprojection_error1, reprojection_coord1 = reprojection_error_in_camera_coord(
            K1[bs], threeD_coords_in_camera1.T, pts1[mask]
        )
        mean_reprojection_error1 = np.mean(reprojection_error1)

        if save_path is not None:
            # used to visulatize and debug
            image_pair_name = [
                osp.splitext(osp.basename(data["pair_names"][0][0]))[0],
                osp.splitext(osp.basename(data["pair_names"][1][0]))[0],
            ]
            file_name = "-".join(image_pair_name) + ".ply"
            save_triangulated_point_cloud(osp.join(save_path, file_name), threeD_coords)

        if debug:
            depth0_gt = (
                data["depth0"][bs][pts0[mask][:, 1], pts0[mask][:, 0]].cpu().numpy()
            )
            nonzero_mask = depth0_gt != 0
            pts0_h_with_depth = (
                np.concatenate([pts0[mask], np.ones((pts0[mask].shape[0], 1))], axis=-1)
                * depth0_gt[:, None]
            )  # N*3
            camera0_gt_point_cloud = (
                np.linalg.inv(K0[bs]) @ pts0_h_with_depth[nonzero_mask].T
            )  # 3*N
            camera0_gt_point_cloud = camera0_gt_point_cloud.T  # N*3

        # T1_gt = data['T_0to1'].cpu().numpy()[bs][:3]
        # triangulated_points_gt_pose = cv2.triangulatePoints(np.dot(K0[bs],T0), np.dot(K1[bs],T1_gt), pts0[mask].T, pts1[mask].T)
        # threeD_coords_gt_pose = cv2.convertPointsFromHomogeneous(triangulated_points_gt_pose.T).squeeze()

        # depth0_gt = data['depth0'][bs][pts0[mask][:,1], pts0[mask][:,0]]

    # used to debug
    # TODO: add bs != 1 scenario and no good initialize boundery scenario
    if debug:
        if reprojection_coord0 is not None and reprojection_coord1 is not None:
            assert reprojection_coord0.shape[0] == reprojection_coord1.shape[0]
            data.update(
                {
                    "reprojected_bids": np.zeros((reprojection_coord0.shape[0],)),
                    "image0_reprojected_initial_keypoints": reprojection_coord0,
                    "image1_reprojected_initial_keypoints": reprojection_coord1,
                }
            )
            data.update({"point_cloud_camera0_gt": camera0_gt_point_cloud})

    data.update(
        {
            "point_cloud_bids": torch.zeros(
                (threeD_coords.shape[0],), device=device
            ),  # NOTE: equal to reprojected bids
            "threeD_init": torch.from_numpy(threeD_coords).to(device),
            "depth0_init": torch.from_numpy(camera0_depth).to(device),
        }
    )  # convert from numpy.array to torch.tensor


@torch.no_grad()
def threeD_points_initialize_DeepLM(
    data,
    LM_config,
    method="reprojection_error",
    triangulate_inlier_only=True,
    debug=False,
    verbose=False,
):
    """
    Use initialized pose to triangulate two-view 3D points to initial depth of image0 
    The world coordinate is defined to camera_0 camear coordinate
    Parameter:
    ----------
    data: dict
    method : str
        choice ['reprojection_method', 'mid_point_method']
    triangulate_inlier_only : bool, optional
        if true : only triangulate inlier keypoints, inliers are extractes in pose initializtion
        else : triangulate all keypoints
    debug : bool, optional
        if true : save intermediate results for dump and visualize

    Update: 
    ----------
    dict: {
        'threeD_init' List[torch.tensor : L*3]: N_bs
        'depth0_init' List[torch.tensor : L] : N_bs
        }
    """
    m_bids = data["m_bids"]
    pts0 = data["mkpts0_f"]
    pts1 = data["mkpts1_f"]
    K0 = data["K0"]
    K1 = data["K1"]
    device = data["m_bids"].device

    if method == "reprojection_method":
        loss_function = triangulation_reprojection_error
    elif method == "mid_point_method":
        loss_function = triangulation_mid_point_error
    else:
        raise NotImplementedError

    # TODO: parallel evaluation
    for bs in range(K0.shape[0]):
        mask = m_bids == bs

        if data["initial_pose"][bs] is not None:
            R, t, inlier_mask = data["initial_pose"][bs]
            mask = mask & inlier_mask if triangulate_inlier_only else mask
        else:
            point_cloud, camera0_depth = None, None
            reprojection_coord0, reprojection_coord1 = None, None
            camera0_gt_point_cloud = None
        T0 = torch.cat(
            [torch.eye(3).to(device), torch.zeros((3, 1)).to(device)], dim=-1
        )
        T1 = torch.cat([R, t], dim=-1)  # 3*4

        # convert to angle_axis(0:3) and translation (3:6) format
        T0, T1 = map(
            lambda a: torch.cat(
                [transforms.so3_log_map(a[:, :3].unsqueeze(0)), a[:, 3].unsqueeze(0)],
                dim=-1,
            ),
            [T0, T1],
        )  # 1*6

        # build DeepLM optimization format
        num_kpts = mask.int().sum()
        T = torch.cat(
            [T0.expand(num_kpts, -1), T1.expand(num_kpts, -1)], dim=0
        ).double()  # 2L*6 build constants format
        K = torch.cat(
            [
                K0[bs].unsqueeze(0).expand(num_kpts, -1, -1),
                K1[bs].unsqueeze(0).expand(num_kpts, -1, -1),
            ],
            dim=0,
        ).double()  # 2L*3*3
        indices = torch.arange(0, num_kpts, device=device).repeat(2)  # build index

        # point_cloud = torch.randn((num_kpts, 3), dtype=torch.float64, device=device) # point cloud initialization L*3 all one
        point_cloud = (
            torch.ones((num_kpts, 3), dtype=torch.float64, device=device) * 2
            if "threeD_init" not in data
            else data["threeD_init"].double()
        )  # point cloud initialization L*3 all one
        # point_cloud = data['threeD_init'].double()

        with torch.enable_grad():
            if verbose:
                logger.info("Triangulation begin......")
            Solve(
                variables=[point_cloud],
                constants=[T, torch.cat([pts0[mask], pts1[mask]], dim=0), K],
                indices=[indices],
                fn=loss_function,
                optimization_cfgs=LM_config,
                verbose=verbose,
            )

        point_cloud = point_cloud.float()

        camera0_depth = (
            (K0[bs] @ point_cloud.transpose(1, 0))[-1].view(-1, 1).squeeze(1)
        )

        if debug:
            # calculate camera0 reprojection error
            (
                reprojection_error0,
                reprojection_coord0,
            ) = reprojection_error_in_camera_coord(K0[bs], point_cloud, pts0[mask])
            mean_reprojection_error0 = torch.mean(reprojection_error0)

            # calculate camera1 reprojection error
            threeD_coords_in_camera1 = R @ point_cloud.transpose(1, 0) + t
            (
                reprojection_error1,
                reprojection_coord1,
            ) = reprojection_error_in_camera_coord(
                K1[bs], threeD_coords_in_camera1.T, pts1[mask]
            )
            mean_reprojection_error1 = torch.mean(reprojection_error1)

            depth0_gt = data["depth0"][bs][
                pts0[mask][:, 1].long(), pts0[mask][:, 0].long()
            ]
            nonzero_mask = depth0_gt != 0
            pts0_h_with_depth = (
                torch.cat(
                    [pts0[mask], torch.ones((pts0[mask].shape[0], 1), device=device)],
                    axis=-1,
                )
                * depth0_gt[:, None]
            )  # N*3
            camera0_gt_point_cloud = torch.inverse(K0[bs]) @ pts0_h_with_depth[
                nonzero_mask
            ].transpose(
                1, 0
            )  # 3*N
            camera0_gt_point_cloud = camera0_gt_point_cloud.transpose(1, 0)  # N*3

    # used to debug
    # TODO: add bs != 1 scenario and no good initialize boundery scenario
    if debug:
        if reprojection_coord0 is not None and reprojection_coord1 is not None:
            assert reprojection_coord0.shape[0] == reprojection_coord1.shape[0]
            data.update(
                {
                    "reprojected_bids": torch.zeros(
                        (reprojection_coord0.shape[0],), device=device
                    ),
                    "image0_reprojected_initial_keypoints": reprojection_coord0,
                    "image1_reprojected_initial_keypoints": reprojection_coord1,
                }
            )
            data.update({"point_cloud_camera0_gt": camera0_gt_point_cloud})

    data.update(
        {
            "point_cloud_bids": torch.zeros((point_cloud.shape[0],), device=device),
            "threeD_init": point_cloud,
            "depth0_init": camera0_depth,
        }
    )  # convert from numpy.array to torch.tensor


@torch.no_grad()
def threeD_points_initialize_known_depth(data, inlier_only=True):
    """
    Use known depth as initialization and only use these points to optimization
    """
    assert "keypoints0_previous" in data and "depth_previous" in data
    m_bids = data["m_bids"]
    pts0 = data["mkpts0_f"]
    pts1 = data["mkpts1_f"]
    K0 = data["K0"]
    K1 = data["K1"]
    pts0_previous = data["keypoints0_previous"]
    depth_previous = data["depth_previous"]
    device = pts0.device

    for bs in range(K0.shape[0]):
        mask = m_bids == bs

        if data["initial_pose"][bs] is not None:
            R, t, inlier_mask = data["initial_pose"][bs]
            # mask = mask & inlier_mask if inlier_only else mask
        else:
            mask = None

        # TODO: Debug here
        _, index_overlap_kpts0, index_overlap_kpts_previous = findOverlapIndex(
            pts0_previous, pts0[mask]
        )

        assert (
            index_overlap_kpts_previous.max() <= depth_previous.shape[0] - 1
        ), f"index over flow! max index: {index_overlap_kpts_previous.max()}, max number: {depth_previous.shape[0]-1}"
        depth_initial_from_previous = depth_previous[index_overlap_kpts_previous]

        if inlier_only and data["initial_pose"][bs] is not None:
            _, _, inlier_mask = data["initial_pose"][bs]

            # Select inlier keypoints with previsouly known depth
            assert (
                index_overlap_kpts0.max() <= inlier_mask.shape[0] - 1
            ), f"index overflow! max index {index_overlap_kpts0.max()}, max number: {depth_previous.shape[0] - 1}"
            previous_depth_inlier_mask = inlier_mask[index_overlap_kpts0]
            depth_initial_from_previous = depth_initial_from_previous[
                previous_depth_inlier_mask
            ]

            # Update keypoints mask:
            # TODO: check weither data dict is updated outside?
            known_depth_mask = torch.zeros((inlier_mask.shape[0],), device=device).bool() # N
            known_depth_mask[index_overlap_kpts0] = True
            data["initial_pose"][bs][2] = inlier_mask & known_depth_mask
        else:
            raise NotImplementedError

    # TODO: add batch operation
    data.update(
        {
            "depth0_init": depth_initial_from_previous.squeeze(-1), # N
            "point_cloud_bids": torch.zeros(
                (depth_initial_from_previous.shape[0],), device=device
            ),
        }
    )


def reprojection_error_in_camera_coord(K, threeD_points_in_camera_coord, twoD_points):
    """
        input:
            K : [3*3],
            threeD_points_in_camera_coord : [N*3], Note: threeD_points are tranformed to camera coordinate,
            twoD_points : [N*2],
        return:
            reprojection_error : [N*2]
            reprojection_coords : [N*2]
        """
    reprojection_coord_h = (
        np.dot(K, threeD_points_in_camera_coord.T)
        if isinstance(threeD_points_in_camera_coord, np.ndarray)
        else K @ threeD_points_in_camera_coord.transpose(1, 0)
    )
    reprojection_coord = reprojection_coord_h[:2] / (reprojection_coord_h[2] + 1e-4)
    if isinstance(reprojection_coord, np.ndarray):
        reprojection_coord = reprojection_coord.T
        reprojection_error = np.linalg.norm(twoD_points - reprojection_coord, axis=1)
    else:
        reprojection_coord = reprojection_coord.transpose(1, 0)
        reprojection_error = torch.linalg.norm(twoD_points - reprojection_coord, dim=-1)
    return reprojection_error, reprojection_coord


def save_triangulated_point_cloud(path, point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if not osp.exists(path.rsplit("/", 1)[0]):
        os.makedirs(path.rsplit("/", 1)[0])
    o3d.io.write_point_cloud(path, pcd)


# NOTE: from huangdi written by shuaiqing, maybe useful in future maybe not. Maybe faster than optimization method
def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1] > 0).sum(axis=0)
    if keypoints_pre is None:
        valid_joint = np.where(v > 1)[0]
    else:
        valid_joint = np.where(v > 0)[0]
    if len(valid_joint) < 1:
        result = np.zeros((keypoints_.shape[1], 4))
        return result
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0) / v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    # ATTN: use middle point
    result[:, :3] = X[:, :3].mean(axis=0, keepdims=True)
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result
