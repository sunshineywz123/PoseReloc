import torch
import torch.nn.functional as f

from .residual_utils import (
    AngleAxisRotatePoint,
    coord_normalization,
    sample_feature_from_unfold_featuremap,
)


def pose_optimization_residual(
    initial_pose,
    mkpts_3D,
    mkpts_query_c,
    mkpts_query_f,
    distance_map,
    intrinsic,
    scale=None,
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
    initial_pose: torch.tensor L*6 or L*1*6
    intrinsic: torch.tensor 3*3
    mkpts_3D: L*2
    mkpts_query_c: L*2
    mkpts_query_f: L*2
    distance_map: L*WW*1
    scale: L*2
    confidance: L*1
    """

    # Dim check
    initial_pose = (
        initial_pose.squeeze(1) if len(initial_pose.shape) == 3 else initial_pose
    )
    mkpts_3D = mkpts_3D.squeeze(1) if len(mkpts_3D.shape) == 3 else mkpts_3D
    mkpts_query_c = (
        mkpts_query_c.squeeze(1) if len(mkpts_query_c.shape) == 3 else mkpts_query_c
    )
    mkpts_query_f = (
        mkpts_query_f.squeeze(1) if len(mkpts_query_f.shape) == 3 else mkpts_query_f
    )

    intrinsic = intrinsic.squeeze(1) if len(intrinsic.shape) == 4 else intrinsic
    distance_map = (
        distance_map.squeeze(1) if len(distance_map.shape) == 4 else distance_map
    )

    device = mkpts_3D.device

    # Rotation and translation
    mkpts_3D_cam = (
        AngleAxisRotatePoint(initial_pose[:, :3], mkpts_3D)
        + initial_pose[:, 3:6]
    ).transpose(
        1, 0
    )  # (3, L)

    # Projection
    mkpts_frame_h = (intrinsic @ mkpts_3D_cam).transpose(1, 0)  # (N*3)
    mkpts_frame = mkpts_frame_h[:, :2] / (mkpts_frame_h[:, [2]] + 1e-4)

    if mode == "feature_metric_error":
        distance, outof_grid_mask = sample_feature_from_unfold_featuremap(
            distance_map,
            offset=mkpts_frame - mkpts_query_c,
            scale=scale,
            mode="Offset_Sample",
            return_outof_grid_mask=True,
            verbose=verbose,
        )  # distance: N*1
        # TODO: solve scale problem, mkpts is in original image resolution, scale should to be original image to fine scale

        if enable_feature_distance_loss:
            out_of_grid_scale = distance_loss_scale

            center_distance = mkpts_frame - mkpts_query_c
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
        distance = mkpts_frame - mkpts_query_f
        distance = distance
    else:
        raise NotImplementedError

    if confidance is not None:
        return distance[confidance > 0], confidance
    else:
        return distance, confidance