import torch
import torch.nn.functional as f
from pytorch3d import transforms

from .residual_utils import (AngleAxisRotatePoint, coord_normalization,
                    sample_feature_from_unfold_featuremap)


def residual(
    depth,
    intrinsic0,
    intrinsic1,
    pose,
    mkpts0_c,
    mkpts1_c,
    mkpts1_f,
    distance_map,
    scale0=None,
    scale1=None,
    enable_feature_distance_loss=True,
    distance_loss_scale=80,
    mode="feature_metric_error",
    verbose=True,
    **kwargs
):
    """
    Parameters:
    -------------
    depth: torch.tensor L*1 (variable) 
    pose: torch.tensor L*6 or L*1*6
    intrinsic0: torch.tensor L*3*3
    intrinsic1: torch.tensor L*3*3
    mkpts0_c: L*2
    mkpts1_c: L*2
    mkpts1_f: L*2
    distance_map: L*WW*1
    scale: L*2
    """

    # Dim check
    depth = depth.squeeze(1) if len(depth.shape) == 3 else depth
    pose = pose.squeeze(1) if len(pose.shape) == 3 else pose
    mkpts0_c = mkpts0_c.squeeze(1) if len(mkpts0_c.shape) == 3 else mkpts0_c
    mkpts1_c = mkpts1_c.squeeze(1) if len(mkpts1_c.shape) == 3 else mkpts1_c
    mkpts1_f = mkpts1_f.squeeze(1) if len(mkpts1_f.shape) == 3 else mkpts1_f

    intrinsic0 = intrinsic0.squeeze(1) if len(intrinsic0.shape) == 4 else intrinsic0
    intrinsic1 = intrinsic1.squeeze(1) if len(intrinsic1.shape) == 4 else intrinsic1
    distance_map = (
        distance_map.squeeze(1) if len(distance_map.shape) == 4 else distance_map
    )

    device = depth.device

    # Unproject
    kpts0_h = (
        torch.cat([mkpts0_c, torch.ones((mkpts0_c.shape[0], 1), device=device)], dim=-1)
        * depth
    )  # (N, 3)
    kpts0_cam = intrinsic0.inverse() @ kpts0_h.unsqueeze(-1)  # (N*3*1)

    # Rotation and translation
    w_kpts0_cam = (
        AngleAxisRotatePoint(pose[:, :3], kpts0_cam.squeeze(-1)) + pose[:, 3:6]
    )  # (N*3)

    # Projection
    w_kpts0_frame_h = (intrinsic1 @ w_kpts0_cam.unsqueeze(-1)).squeeze(-1)  # (N*3)
    w_kpts0_frame = w_kpts0_frame_h[:, :2] / (w_kpts0_frame_h[:, [2]] + 1e-4)

    if mode == "feature_metric_error":
        distance, outof_grid_mask = sample_feature_from_unfold_featuremap(
            distance_map,
            offset=w_kpts0_frame - mkpts1_c,
            scale=scale1,
            mode="Offset_Sample",
            return_outof_grid_mask=True,
            verbose=verbose
        )  # distance: N*1
        # TODO: solve scale problem, mkpts is in original image resolution, scale should to be original image to fine scale

        if enable_feature_distance_loss:
            out_of_grid_scale = distance_loss_scale

            center_distance = w_kpts0_frame - mkpts1_c
            center_distance = torch.linalg.norm(center_distance, dim=-1).unsqueeze(
                1
            )  # L*1
            # if reprojected points locat in grid, distance loss are 0
            weight = (
                torch.ones_like(center_distance, device=center_distance.device)
                * out_of_grid_scale
            )
            weight[~outof_grid_mask] = 0
            # center_distance[~outof_grid_mask] = torch.clamp(center_distance[~outof_grid_mask], 0,0)
            # center_distance[~outof_grid_mask] = 0
            center_distance *= weight.detach()

            distance = torch.cat([distance, center_distance], dim=1)

    elif mode == "geometry_error":
        distance = w_kpts0_frame - mkpts1_f
    else:
        raise NotImplementedError

    return distance


def pose_ba_residual(
    pose0,
    pose1,
    depth,
    intrinsic0,
    intrinsic1,
    mkpts0_c,
    mkpts1_c,
    mkpts1_f,
    distance_map,
    scale0=None,
    scale1=None,
    enable_feature_distance_loss=True,
    distance_loss_scale=80,
    mode="feature_metric_error",
    confidance=None,
    verbose=True,
    **kwargs
):
    """
    Parameters:
    -------------
    pose0: torch.tensor L*6 or L*1*6
    pose1: torch.tensor L*6 or L*1*6
    depth: torch.tensor L*1 (variable) 
    intrinsic0: torch.tensor L*3*3
    intrinsic1: torch.tensor L*3*3
    mkpts0_c: L*2
    mkpts1_c: L*2
    mkpts1_f: L*2
    distance_map: L*WW*1
    scale: L*2
    confidance: L*1
    """

    # Dim check
    depth = depth.squeeze(1) if len(depth.shape) == 3 else depth
    pose0 = pose0.squeeze(1) if len(pose0.shape) == 3 else pose0
    pose1 = pose1.squeeze(1) if len(pose1.shape) == 3 else pose1
    mkpts0_c = mkpts0_c.squeeze(1) if len(mkpts0_c.shape) == 3 else mkpts0_c
    mkpts1_c = mkpts1_c.squeeze(1) if len(mkpts1_c.shape) == 3 else mkpts1_c
    mkpts1_f = mkpts1_f.squeeze(1) if len(mkpts1_f.shape) == 3 else mkpts1_f

    intrinsic0 = intrinsic0.squeeze(1) if len(intrinsic0.shape) == 4 else intrinsic0
    intrinsic1 = intrinsic1.squeeze(1) if len(intrinsic1.shape) == 4 else intrinsic1
    distance_map = (
        distance_map.squeeze(1) if len(distance_map.shape) == 4 else distance_map
    )

    device = depth.device

    # Unproject
    kpts0_h = (
        torch.cat([mkpts0_c, torch.ones((mkpts0_c.shape[0], 1), device=device)], dim=-1)
        * depth
    )  # (N, 3)
    kpts0_cam0 = intrinsic0.inverse() @ kpts0_h.unsqueeze(-1)  # (N*3*1)

    # Rotation and translation
    # inverse pose0
    R_inverse = transforms.so3_exponential_map(pose0[:, :3]).inverse() # (N*3*3)
    t_inverse = -1 * (R_inverse @ pose0[:, 3:6].unsqueeze(-1)).squeeze(-1) # N*3
    angle_axis_inverse = transforms.so3_log_map(R_inverse)
    pose0_inverse = torch.cat([angle_axis_inverse, t_inverse], dim=1) # N*6

    w_kpts0_world = (
        AngleAxisRotatePoint(pose0_inverse[:, :3], kpts0_cam0.squeeze(-1)) + pose0_inverse[:, 3:6]
    )  # (N*3)
    w_kpts0_cam1 = (
        AngleAxisRotatePoint(pose1[:, :3], w_kpts0_world.squeeze(-1)) + pose1[:, 3:6]
    )  # (N*3)

    # Projection
    w_kpts0_frame1_h = (intrinsic1 @ w_kpts0_cam1.unsqueeze(-1)).squeeze(-1)  # (N*3)
    w_kpts0_frame1 = w_kpts0_frame1_h[:, :2] / (w_kpts0_frame1_h[:, [2]] + 1e-4)

    if mode == "feature_metric_error":
        distance, outof_grid_mask = sample_feature_from_unfold_featuremap(
            distance_map,
            offset=w_kpts0_frame1 - mkpts1_c,
            scale=scale1,
            mode="Offset_Sample",
            return_outof_grid_mask=True,
            verbose=verbose
        )  # distance: N*1
        # TODO: solve scale problem, mkpts is in original image resolution, scale should to be original image to fine scale

        if enable_feature_distance_loss:
            out_of_grid_scale = distance_loss_scale

            center_distance = w_kpts0_frame1 - mkpts1_c
            center_distance = torch.linalg.norm(center_distance, dim=-1).unsqueeze(
                1
            )  # L*1
            # if reprojected points locat in grid, distance loss are 0
            weight = (
                torch.ones_like(center_distance, device=center_distance.device)
                * out_of_grid_scale
            )
            weight[~outof_grid_mask] = 0
            # center_distance[~outof_grid_mask] = torch.clamp(center_distance[~outof_grid_mask], 0,0)
            # center_distance[~outof_grid_mask] = 0
            center_distance *= weight.detach()

            distance = torch.cat([distance, center_distance], dim=1)
            if confidance is not None:
                confidance[outof_grid_mask] -= 1

    elif mode == "geometry_error":
        distance = w_kpts0_frame1 - mkpts1_f
    else:
        raise NotImplementedError

    if confidance is not None:
        return distance[confidance > 0], confidance
    else:
        return distance, confidance

def depth_residual(
    depth,
    pose0,
    pose1,
    intrinsic0,
    intrinsic1,
    mkpts0_c,
    mkpts1_c,
    mkpts1_f,
    distance_map,
    scale0=None,
    scale1=None,
    enable_feature_distance_loss=True,
    distance_loss_scale=80,
    mode="feature_metric_error",
    confidance=None,
    verbose=True,
    **kwargs
):
    """
    Parameters:
    -------------
    pose0: torch.tensor L*6 or L*1*6
    pose1: torch.tensor L*6 or L*1*6
    depth: torch.tensor L*1 (variable) 
    intrinsic0: torch.tensor L*3*3
    intrinsic1: torch.tensor L*3*3
    mkpts0_c: L*2
    mkpts1_c: L*2
    mkpts1_f: L*2
    distance_map: L*WW*1
    scale: L*2
    confidance: L*1
    """

    # Dim check
    depth = depth.squeeze(1) if len(depth.shape) == 3 else depth
    pose0 = pose0.squeeze(1) if len(pose0.shape) == 3 else pose0
    pose1 = pose1.squeeze(1) if len(pose1.shape) == 3 else pose1
    mkpts0_c = mkpts0_c.squeeze(1) if len(mkpts0_c.shape) == 3 else mkpts0_c
    mkpts1_c = mkpts1_c.squeeze(1) if len(mkpts1_c.shape) == 3 else mkpts1_c
    mkpts1_f = mkpts1_f.squeeze(1) if len(mkpts1_f.shape) == 3 else mkpts1_f

    intrinsic0 = intrinsic0.squeeze(1) if len(intrinsic0.shape) == 4 else intrinsic0
    intrinsic1 = intrinsic1.squeeze(1) if len(intrinsic1.shape) == 4 else intrinsic1
    distance_map = (
        distance_map.squeeze(1) if len(distance_map.shape) == 4 else distance_map
    )

    device = depth.device

    # Unproject
    kpts0_h = (
        torch.cat([mkpts0_c, torch.ones((mkpts0_c.shape[0], 1), device=device)], dim=-1)
        * depth
    )  # (N, 3)
    kpts0_cam0 = intrinsic0.inverse() @ kpts0_h.unsqueeze(-1)  # (N*3*1)

    # Rotation and translation
    # inverse pose0
    # FIXME: check gredient problem
    R_inverse = transforms.so3_exponential_map(pose0[:, :3]).inverse() # (N*3*3)
    t_inverse = -1 * (R_inverse @ pose0[:, 3:6].unsqueeze(-1)).squeeze(-1) # N*3
    angle_axis_inverse = transforms.so3_log_map(R_inverse)
    pose0_inverse = torch.cat([angle_axis_inverse, t_inverse], dim=1) # N*6

    w_kpts0_world = (
        AngleAxisRotatePoint(pose0_inverse[:, :3], kpts0_cam0.squeeze(-1)) + pose0_inverse[:, 3:6]
    )  # (N*3)
    w_kpts0_cam1 = (
        AngleAxisRotatePoint(pose1[:, :3], w_kpts0_world.squeeze(-1)) + pose1[:, 3:6]
    )  # (N*3)

    # Projection
    w_kpts0_frame1_h = (intrinsic1 @ w_kpts0_cam1.unsqueeze(-1)).squeeze(-1)  # (N*3)
    w_kpts0_frame1 = w_kpts0_frame1_h[:, :2] / (w_kpts0_frame1_h[:, [2]] + 1e-4)

    if mode == "feature_metric_error":
        distance, outof_grid_mask = sample_feature_from_unfold_featuremap(
            distance_map,
            offset=w_kpts0_frame1 - mkpts1_c,
            scale=scale1,
            mode="Offset_Sample",
            return_outof_grid_mask=True,
            verbose=verbose
        )  # distance: N*1
        # TODO: solve scale problem, mkpts is in original image resolution, scale should to be original image to fine scale

        if enable_feature_distance_loss:
            out_of_grid_scale = distance_loss_scale

            center_distance = w_kpts0_frame1 - mkpts1_c
            center_distance = torch.linalg.norm(center_distance, dim=-1).unsqueeze(
                1
            )  # L*1
            # if reprojected points locat in grid, distance loss are 0
            weight = (
                torch.ones_like(center_distance, device=center_distance.device)
                * out_of_grid_scale
            )
            weight[~outof_grid_mask] = 0
            # center_distance[~outof_grid_mask] = torch.clamp(center_distance[~outof_grid_mask], 0,0)
            # center_distance[~outof_grid_mask] = 0
            center_distance *= weight.detach()

            distance = torch.cat([distance, center_distance], dim=1)
            if confidance is not None:
                confidance[outof_grid_mask] -= 1

    elif mode == "geometry_error":
        distance = w_kpts0_frame1 - mkpts1_f
    else:
        raise NotImplementedError

    if confidance is not None:
        return distance[confidance > 0], confidance
    else:
        return distance, confidance