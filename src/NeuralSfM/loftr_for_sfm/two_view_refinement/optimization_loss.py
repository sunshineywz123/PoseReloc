import torch
import torch.nn.functional as F
from pytorch3d import transforms
import math

from .utils import (
    AngleAxisRotatePoint,
    sample_feature_from_featuremap,
    sample_feature_from_unfold_featuremap,
)


def Two_view_featuremetric_error(
    depth,
    pose,
    kpts0,
    kpts1,
    feature0,
    feature_map1,
    K0,
    K1,
    img0_hw=None,
    img1_hw=None,
    scale=None,
    norm_feature=False,
    residual_format="L*1",
    use_angle_axis=True,
    use_fine_unfold_feature=False,
    enable_center_distance_loss=False,
    distance_loss_scale=1000,
    return_reprojected_coord=False,  # used for debug, can't be used for DeepLM
    tragetory_dict=None,
    patch_size=None,
    **kwargs,
):
    """
    Two view featuremetric error for refine depth of image0 or BA
    NOTE: not implement batch operation

    Parameter:
    -----------
    depth : torch.tensor L*1,
    pose : torch.tensor L*6 or L*1*6
        one camera pose, initialized relative pose from camera0 to camera1, L post <-> L points
    kpts0 : torch.tensor L*2 (original image resolution),
    kpts1 : torch.tensor L*2 (original image resolution),
        note that if use_unfold_feature, kpts0 & kpts1 should correspond to centeral point of local window feature
    NOTE: following paras are bs=1 temporaryly
    K0 : torch.tensor 3*3,
    K1 : torch.tensor 3*3,
    feature0 : torch.tensor L*c(sample feature0 in advance beacuse keypoints0 are fixed) or L*W*W*c
    feature_map1 : torch.tensor c*h*w or L*ww*c(fine unfold feature)
    img0_hw : torch.tensor 2 
        note that only useful when use_fine_unfold_feature=False
    img1_hw : torch.tensor 2
        note that only useful when use_fine_unfold_feature=False
    scale : input_size / fine_level_size
        note that only useful when use_fine_unfold_feature=True
    norm_feature : bool
    residual_format : str
        choice: ['L*1', '1*1]
    use_angle_axis: bool 
        Two implement methods to transform 3D points.
        If true: use angle axis vector to convert 3D points
        else: transform angle axis vector to R, and then use R matrix to convert 3D points
    enable_center_distance_loss : bool
        used to constrains reprojected points inside local grid
    distance_scale : float
        amplifier scale for distance loss
    return_reprojected_coord : bool
        whether return reprojected coordinate   
    tragetory_dict : dict
        a hook to get reprojected keypoints for outside
    patch_size : int
        patch alignment size, None is not use patch alignment

    Return:
    -----------
    residual : torch.tensor L*1 or L*C

    """
    # Dim check
    depth = depth.squeeze(1) if len(depth.shape) == 3 else depth
    pose = pose.squeeze(1) if len(pose.shape) == 3 else pose
    kpts0 = kpts0.squeeze(1) if len(kpts0.shape) == 3 else kpts0
    kpts1 = kpts1.squeeze(1) if len(kpts1.shape) == 3 else kpts1

    # Type check: convert to float32 and convert back to float64
    depth, kpts0, kpts1, pose = map(lambda a: a.float(), [depth, kpts0, kpts1, pose])

    # feature0 not sampled in advance and sample in the process of optimization code:
    # device = feature_map0.device
    # # Get features for keypoints0 in image0
    # if not use_fine_unfold_feature:
    #     feature0 = sample_feature_from_featuremap(
    #         feature_map0, kpts0, img0_hw, norm_feature=norm_feature
    #     )
    # else:
    #     if feature_map0.shape[0] != kpts0.shape[0]:
    #         assert feature_map0.shape[0] > kpts0.shape[0]
    #         feature_map0 = feature_map0[: kpts0.shape[0]]
    #     feature0 = sample_feature_from_unfold_featuremap(
    #         feature_map0, norm_feature=norm_feature
    #     )

    device = feature0.device
    # feature0 are sampled in advance
    feature0 = feature0[: kpts0.shape[0]]
    feature0 = F.normalize(feature0, p=2, dim=-1) if norm_feature else feature0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones((kpts0.shape[0], 1), device=device)], dim=-1)
        * depth
    )  # (N, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(1, 0)  # (3, L)

    if use_angle_axis:
        w_kpts0_cam = (
            AngleAxisRotatePoint(pose[:, :3], kpts0_cam.transpose(1, 0)) + pose[:, 3:6]
        ).transpose(
            1, 0
        )  # (3, L)
    else:
        R = transforms.so3_exponential_map(pose[:, :3])  # L*3*3
        w_kpts0_cam = (
            (R @ kpts0_cam.transpose(1, 0).unsqueeze(-1) + pose[:, 3:6].unsqueeze(-1))
            .squeeze()
            .transpose(1, 0)
        )  # (3, L)
    w_kpts0_depth_computed = w_kpts0_cam[2, :]  # z

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(1, 0)  # (L, 3)
    w_kpts0 = w_kpts0_h[:, :2] / (
        w_kpts0_h[:, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # TODO: add Covisible Check
    # Get features for reprojected keypoints in image1
    if not use_fine_unfold_feature:
        feature1 = sample_feature_from_featuremap(
            feature_map1,
            w_kpts0,
            img1_hw,
            norm_feature=norm_feature,
            patch_feature_size=patch_size,
        )
        center_distance = None
    else:
        if feature_map1.shape[0] != kpts1.shape[0]:
            assert feature_map1.shape[0] > kpts1.shape[0]
            feature_map1 = feature_map1[: kpts1.shape[0]]
        feature1, outof_grid_mask = sample_feature_from_unfold_featuremap(
            feature_map1,
            offset=w_kpts0 - kpts1,
            scale=scale,
            mode="Offset_Sample",
            norm_feature=norm_feature,
            return_outof_grid_mask=True,
            patch_feature_size=patch_size,
        )

        if enable_center_distance_loss:
            # #Used to substitute distance_loss_scale
            # M, WW, C = feature_map1.shape
            # W = int(math.sqrt(WW))
            # feature_distance = torch.linalg.norm(feature1 - feature0, dim=-1).unsqueeze(1)  # L*1
            # out_of_grid_scale = feature_distance / (scale * W)

            out_of_grid_scale = distance_loss_scale

            center_distance = w_kpts0 - kpts1
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

    # grads = {}
    # def save_grad():
    #     def hook(grad):
    #         grads.update({"w_kpts0_grad" : grad})
    #         print(grad)
    #     return hook
    # if w_kpts0.requires_grad == True:
    #     # only useful when called by backward!
    #     w_kpts0.register_hook(save_grad())
    if len(feature0.shape) != 2:
        # local patch alignment scenario
        # TODO: check whether it's a good inplementation
        assert len(feature0.shape) == len(feature1.shape)
        num = feature0.shape[0]
        feature0 = feature0.reshape(num, -1)
        feature1 = feature1.reshape(num, -1)

    if residual_format == "L*1":
        distance = torch.linalg.norm(feature1 - feature0, dim=-1).unsqueeze(1)  # L*1
    elif residual_format == "L*C":
        distance = feature1 - feature0
    else:
        raise NotImplementedError

    if enable_center_distance_loss:
        distance = (
            torch.cat([distance, center_distance], dim=1)
            if center_distance is not None
            else distance
        )

    if tragetory_dict is not None and kpts1.shape[0] != 1:
        # For second-order optimizer DeepLM, True: append; False: not append
        if "marker" in kwargs:
            tragetory_list = tragetory_dict["w_kpts0_list"]
            tragetory_list[-1].append(w_kpts0.clone().detach()) if kwargs[
                "marker"
            ] == True else None
            tragetory_dict.update({"w_kpts0_list": tragetory_list})
    if return_reprojected_coord:
        if "marker_return" in kwargs:
            # For first-order optimizer
            if kwargs["marker_return"] == True:
                return distance.double(), w_kpts0
    return distance.double()
    # return torch.mean(distance)


def Two_view_featuremetric_error_pose(
    pose,
    depth,
    kpts0,
    kpts1,
    feature0,
    feature_map1,
    K0,
    K1,
    img0_hw=None,
    img1_hw=None,
    scale=None,
    norm_feature=False,
    residual_format="L*1",
    use_angle_axis=True,
    use_fine_unfold_feature=False,
    enable_center_distance_loss=False,
    distance_loss_scale=1000,
    return_reprojected_coord=False,  # used for debug, can't be used for DeepLM
    tragetory_dict=None,
    patch_size=None,
    **kwargs,
):
    """
    Two view featuremetric error for refine pose.
    NOTE: not implement batch operation

    Parameter:
    -----------
        pose : torch.tensor L*6 or L*1*6 or 1*6
        initialized relative pose from camera0 to camera1, L post <-> L points
    depth : torch.tensor L*1,
    kpts0 : torch.tensor L*2 (original image resolution),
    kpts1 : torch.tensor L*2 (original image resolution),
        note that if use_unfold_feature, kpts0 & kpts1 should correspond to centeral point of local window feature
    NOTE: following paras are bs=1 temporaryly
    K0 : torch.tensor 3*3,
    K1 : torch.tensor 3*3,
    feature0 : torch.tensor L*c(sample feature0 in advance beacuse keypoints0 are fixed)
    feature_map1 : torch.tensor c*h*w or L*ww*c(fine unfold feature)
    img0_hw : torch.tensor 2 
        note that only useful when use_fine_unfold_feature=False
    img1_hw : torch.tensor 2
        note that only useful when use_fine_unfold_feature=False
    scale : input_size / fine_level_size
        note that only useful when use_fine_unfold_feature=True
    norm_feature : bool
    residual_format : str
        choice: ['L*1', '1*1]
    use_angle_axis: bool 
        Two implement methods to transform 3D points.
        If true: use angle axis vector to convert 3D points
        else: transform angle axis vector to R, and then use R matrix to convert 3D points
    enable_center_distance_loss : bool
        used to constrains reprojected points inside local grid
    distance_scale : float
        amplifier scale for distance loss
    return_reprojected_coord : bool
        whether return reprojected coordinate   
    tragetory_dict : dict
        a hook to get reprojected keypoints for outside
    patch_size : int
        patch alignment size, None is not use patch alignment

    Return:
    -----------
    residual : torch.tensor L*1 or L*C

    """
    # Dim check and squeeze dim=1 to avoid [L,1] indices
    pose = pose.squeeze(1) if len(pose.shape) == 3 else pose
    depth = depth.squeeze(1) if len(depth.shape) == 3 else depth
    kpts0 = kpts0.squeeze(1) if len(kpts0.shape) == 3 else kpts0
    kpts1 = kpts1.squeeze(1) if len(kpts1.shape) == 3 else kpts1

    # Type check: convert to float32 and convert back to float64
    depth, kpts0, pose = map(lambda a: a.float(), [depth, kpts0, pose])

    # Type check: convert to float32 and convert back to float64
    depth, kpts0, kpts1, pose = map(lambda a: a.float(), [depth, kpts0, kpts1, pose])

    # feature0 not sampled in advance and sample in the process of optimization code:
    # device = feature_map0.device
    # # Get features for keypoints0 in image0
    # if not use_fine_unfold_feature:
    #     feature0 = sample_feature_from_featuremap(
    #         feature_map0, kpts0, img0_hw, norm_feature=norm_feature
    #     )
    # else:
    #     if feature_map0.shape[0] != kpts0.shape[0]:
    #         assert feature_map0.shape[0] > kpts0.shape[0]
    #         feature_map0 = feature_map0[: kpts0.shape[0]]
    #     feature0 = sample_feature_from_unfold_featuremap(
    #         feature_map0, norm_feature=norm_feature
    #     )

    device = feature0.device
    # feature0 are sampled in advance
    feature0 = feature0[: kpts0.shape[0]]
    feature0 = F.normalize(feature0, p=2, dim=-1) if norm_feature else feature0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones((kpts0.shape[0], 1), device=device)], dim=-1)
        * depth
    )  # (N, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(1, 0)  # (3, L)

    if use_angle_axis:
        w_kpts0_cam = (
            AngleAxisRotatePoint(pose[:, :3], kpts0_cam.transpose(1, 0)) + pose[:, 3:6]
        ).transpose(
            1, 0
        )  # (3, L)
    else:
        R = transforms.so3_exponential_map(pose[:, :3])  # L*3*3
        w_kpts0_cam = (
            (R @ kpts0_cam.transpose(1, 0).unsqueeze(-1) + pose[:, 3:6].unsqueeze(-1))
            .squeeze()
            .transpose(1, 0)
        )  # (3, L)
    w_kpts0_depth_computed = w_kpts0_cam[2, :]  # z

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(1, 0)  # (L, 3)
    w_kpts0 = w_kpts0_h[:, :2] / (
        w_kpts0_h[:, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # TODO: add Covisible Check
    # Get features for reprojected keypoints in image1
    if not use_fine_unfold_feature:
        feature1 = sample_feature_from_featuremap(
            feature_map1, w_kpts0, img1_hw, norm_feature=norm_feature, patch_feature_size=patch_size
        )
        center_distance = None
    else:
        if feature_map1.shape[0] != kpts1.shape[0]:
            assert feature_map1.shape[0] > kpts1.shape[0]
            feature_map1 = feature_map1[: kpts1.shape[0]]
        feature1, outof_grid_mask = sample_feature_from_unfold_featuremap(
            feature_map1,
            offset=w_kpts0 - kpts1,
            scale=scale,
            mode="Offset_Sample",
            norm_feature=norm_feature,
            return_outof_grid_mask=True,
            patch_feature_size=patch_size
        )

        if enable_center_distance_loss:
            # #Used to substitute distance_loss_scale
            # M, WW, C = feature_map1.shape
            # W = int(math.sqrt(WW))
            # feature_distance = torch.linalg.norm(feature1 - feature0, dim=-1).unsqueeze(1)  # L*1
            # out_of_grid_scale = feature_distance / (scale * W)

            out_of_grid_scale = distance_loss_scale

            center_distance = w_kpts0 - kpts1
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

    # grads = {} # need to set for global
    # def save_grad():
    #     def hook(grad):
    #         grads.update({"w_kpts0_grad" : grad})
    #         print(grad)
    #     return hook
    # if w_kpts0.requires_grad == True:
    #     # only useful when called by backward!
    #     w_kpts0.register_hook(save_grad())
    if len(feature0.shape) != 2:
        # local patch alignment scenario
        # TODO: check whether it's a good inplementation
        assert len(feature0.shape) == len(feature1.shape)
        num = feature0.shape[0]
        feature0 = feature0.reshape(num, -1)
        feature1 = feature1.reshape(num, -1)

    if residual_format == "L*1":
        distance = torch.linalg.norm(feature1 - feature0, dim=-1).unsqueeze(1)  # L*1
    elif residual_format == "L*C":
        distance = feature1 - feature0
    else:
        raise NotImplementedError

    if enable_center_distance_loss:
        distance = (
            torch.cat([distance, center_distance], dim=-1)
            if center_distance is not None
            else distance
        )

    if tragetory_dict is not None and kpts1.shape[0] != 1:
        # For second-order optimizer DeepLM, True: append; False: not append
        if "marker" in kwargs:
            tragetory_list = tragetory_dict["w_kpts0_list"]
            tragetory_list[-1].append(w_kpts0.clone().detach()) if kwargs[
                "marker"
            ] == True else None
            tragetory_dict.update({"w_kpts0_list": tragetory_list})
    if return_reprojected_coord:
        if "marker_return" in kwargs:
            # For first-order optimizer
            if kwargs["marker_return"] == True:
                return distance.double(), w_kpts0
    return distance.double()
    # return torch.mean(distance)


def Two_view_reprojection_error(
    depth, pose, kpts0, kpts1, K0, K1, use_angle_axis=True, **kwargs
):
    """
    Two view reprojection error for refine depth of img0 or BA
    NOTE: not implement batch operation

    Parameter:
    -----------
    pose : torch.tensor L*6 or L*1*6 or 1*6
        one camera pose, initialized relative pose from camera0 to camera1, L post <-> L points
    kpts0 : torch.tensor L*2 (original image resolution),
    kpts1 : torch.tensor L*2 (original image resolution),
    NOTE: following paras are bs=1 temporary
    K0 : torch.tensor 3*3,
    K1 : torch.tensor 3*3,

    """
    # Dim check
    depth = depth.squeeze(1) if len(depth.shape) == 3 else depth
    pose = pose.squeeze(1) if len(pose.shape) == 3 else pose
    kpts0 = kpts0.squeeze(1) if len(kpts0.shape) == 3 else kpts0
    kpts1 = kpts1.squeeze(1) if len(kpts1.shape) == 3 else kpts1

    # Type check: convert to float32 and convert back to float64
    depth, kpts0, kpts1, pose = map(lambda a: a.float(), [depth, kpts0, kpts1, pose])

    device = depth.device

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones((kpts0.shape[0], 1), device=device)], dim=-1)
        * depth
    )  # (N, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(1, 0)  # (3, L)

    # Rigid Transform
    # if len(pose.shape) == 3: # L*3*4
    #     w_kpts0_cam = (
    #         pose[:, :, :3] @ kpts0_cam.transpose(1, 0).unsqueeze(-1) + pose[:, :, 3:]
    #     ).squeeze().transpose(1,0)  # L*3*1 Note: for N camera to N kpts and then transform to 3*L
    # elif len(pose.shape) == 2: # 3*4 only one camera
    #     w_kpts0_cam = pose[:, :3] @ kpts0_cam + pose[:, 3:]  # (3, L) Note: for one camera to N kpts
    if use_angle_axis:
        w_kpts0_cam = (
            AngleAxisRotatePoint(pose[:, :3], kpts0_cam.transpose(1, 0)) + pose[:, 3:6]
        ).transpose(
            1, 0
        )  # (3, L)
    else:
        R = transforms.so3_exponential_map(pose[:, :3])  # L*3*3
        w_kpts0_cam = (
            (R @ kpts0_cam.transpose(1, 0).unsqueeze(-1) + pose[:, 3:6].unsqueeze(-1))
            .squeeze()
            .transpose(1, 0)
        )  # (3, L)

    w_kpts0_depth_computed = w_kpts0_cam[2, :]  # z

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(1, 0)  # (L, 3)
    w_kpts0 = w_kpts0_h[:, :2] / (
        w_kpts0_h[:, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    distance = w_kpts0 - kpts1
    return distance.double()
    # return torch.mean(distance)


def Two_view_reprojection_error_for_pose(
    pose, depth, kpts0, kpts1, K0, K1, use_angle_axis=True, **kwargs
):
    """
    Two view reprojection error for refine depth of img0 or BA
    NOTE: not implement batch operation

    Parameter:
    -----------
    depth : torch.tensor L*1,
    pose : torch.tensor L*6 or L*1*6 or 1*6
        one camera pose, initialized relative pose from camera0 to camera1, L post <-> L points
    kpts0 : torch.tensor L*2 (original image resolution),
    kpts1 : torch.tensor L*2 (original image resolution),
    NOTE: following paras are bs=1 temporaryly
    K0 : torch.tensor 3*3,
    K1 : torch.tensor 3*3,

    """
    # Dim check
    pose = pose.squeeze(1) if len(pose.shape) == 3 else pose
    depth = depth.squeeze(1) if len(depth.shape) == 3 else depth
    kpts0 = kpts0.squeeze(1) if len(kpts0.shape) == 3 else kpts0
    kpts1 = kpts1.squeeze(1) if len(kpts1.shape) == 3 else kpts1

    # Type check: convert to float32 and convert back to float64
    depth, kpts0, kpts1, pose = map(lambda a: a.float(), [depth, kpts0, kpts1, pose])

    device = depth.device

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones((kpts0.shape[0], 1), device=device)], dim=-1)
        * depth
    )  # (N, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(1, 0)  # (3, L)

    # Rigid Transform
    # if len(pose.shape) == 3: # L*3*4
    #     w_kpts0_cam = (
    #         pose[:, :, :3] @ kpts0_cam.transpose(1, 0).unsqueeze(-1) + pose[:, :, 3:]
    #     ).squeeze().transpose(1,0)  # L*3*1 Note: for N camera to N kpts and then transform to 3*L
    # elif len(pose.shape) == 2: # 3*4 only one camera
    #     w_kpts0_cam = pose[:, :3] @ kpts0_cam + pose[:, 3:]  # (3, L) Note: for one camera to N kpts
    if use_angle_axis:
        w_kpts0_cam = (
            AngleAxisRotatePoint(pose[:, :3], kpts0_cam.transpose(1, 0)) + pose[:, 3:6]
        ).transpose(
            1, 0
        )  # (3, L)
    else:
        R = transforms.so3_exponential_map(
            pose[:, :3]
        )  # L*3*3 convert axis angle vector to R matrix first
        w_kpts0_cam = (
            (R @ kpts0_cam.transpose(1, 0).unsqueeze(-1) + pose[:, 3:6].unsqueeze(-1))
            .squeeze()
            .transpose(1, 0)
        )  # (3, L)

    w_kpts0_depth_computed = w_kpts0_cam[2, :]  # z

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(1, 0)  # (L, 3)
    w_kpts0 = w_kpts0_h[:, :2] / (
        w_kpts0_h[:, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    distance = w_kpts0 - kpts1
    return distance.double()
    # return torch.mean(distance)


def triangulation_reprojection_error(
    point_cloud, pose, kpts, K, use_angle_axis=True, **kwargs
):
    """
    Two view reprojection error for refine depth of img0 or BA
    NOTE: not implement batch operation

    Parameter:
    -----------
    point_cloud : torch.tensor L*3
    pose : torch.tensor L*6 or L*1*6 or 1*6
        one camera pose, initialized relative pose from camera0 to camera1, L post <-> L points
    kpts0 : torch.tensor L*2 (original image resolution),
    kpts1 : torch.tensor L*2 (original image resolution),
    K : torch.tensor L*3*3,

    """
    # Dim check
    point_cloud = point_cloud.squeeze(1) if len(point_cloud.shape) == 3 else point_cloud
    pose = pose.squeeze(1) if len(pose.shape) == 3 else pose
    kpts = kpts.squeeze(1) if len(kpts.shape) == 3 else kpts
    K = K.squeeze(1) if len(K.shape) == 4 else K

    # Type check: convert to float32 and convert back to float64
    point_cloud, kpts, pose, K = map(lambda a: a.float(), [point_cloud, kpts, pose, K])

    device = point_cloud.device

    if use_angle_axis:
        w_kpts_cam = (
            AngleAxisRotatePoint(pose[:, :3], point_cloud) + pose[:, 3:6]
        )  # (L, 3)
    else:
        R = transforms.so3_exponential_map(pose[:, :3])  # L*3*3
        w_kpts_cam = (
            R @ point_cloud.transpose(1, 0).unsqueeze(-1) + pose[:, 3:6].unsqueeze(-1)
        ).squeeze()  # (L, 3)

    w_kpts_depth_computed = w_kpts_cam[:, 2]  # z

    # Project
    w_kpts_h = (K @ w_kpts_cam.unsqueeze(-1)).squeeze(-1)  # (L, 3)
    w_kpts = w_kpts_h[:, :2] / (
        w_kpts_h[:, [2]] + 1e-4
    )  # (L, 2), +1e-4 to avoid zero depth

    distance = w_kpts - kpts
    return distance.double()
    # return torch.mean(distance)


def triangulation_mid_point_error(
    point_cloud, pose, kpts, K, use_angle_axis=True, **kwargs
):
    """
    Two view mid point error for refine depth of img0 or BA
    NOTE: not implement batch operation

    Parameter:
    -----------
    point_cloud : torch.tensor L*3
    pose : torch.tensor L*6 or L*1*6 or 1*6
        one camera pose, initialized relative pose from camera0 to camera1, L post <-> L points
    kpts0 : torch.tensor L*2 (original image resolution),
    kpts1 : torch.tensor L*2 (original image resolution),
    K : torch.tensor L*3*3,

    """
    # Dim check
    point_cloud = point_cloud.squeeze(1) if len(point_cloud.shape) == 3 else point_cloud
    pose = pose.squeeze(1) if len(pose.shape) == 3 else pose
    kpts = kpts.squeeze(1) if len(kpts.shape) == 3 else kpts
    K = K.squeeze(1) if len(K.shape) == 4 else K

    # Type check: convert to float32 and convert back to float64
    point_cloud, kpts, pose, K = map(lambda a: a.float(), [point_cloud, kpts, pose, K])

    device = point_cloud.device

    if use_angle_axis:
        w_kpts_cam = (
            AngleAxisRotatePoint(pose[:, :3], point_cloud) + pose[:, 3:6]
        )  # (L, 3)
    else:
        R = transforms.so3_exponential_map(pose[:, :3])  # L*3*3
        w_kpts_cam = (
            R @ point_cloud.transpose(1, 0).unsqueeze(-1) + pose[:, 3:6].unsqueeze(-1)
        ).squeeze()  # (L, 3)

    w_kpts_depth_computed = w_kpts_cam[:, 2]  # z

    # Get threeD line(direction):
    kpts_h = torch.cat(
        [kpts, torch.ones((kpts.shape[0], 1), device=device)], dim=-1
    ).unsqueeze(
        -1
    )  # L*3*1
    direction = F.normalize(
        (K.inverse() @ kpts_h).squeeze(-1)
    )  # L*3 direction of the 3D line

    # Get distance from 3D point to 3D camera line, note that 3D points are transformed to camera coord by relative pose
    project_threeD_to_line = (
        direction.unsqueeze(1) @ w_kpts_cam.unsqueeze(-1)
    ).squeeze(-1)
    distance = torch.sqrt(
        torch.pow(torch.norm(w_kpts_cam, dim=-1, keepdim=True), 2)
        - torch.pow(project_threeD_to_line, 2)
    )
    return distance.double()
    # return torch.mean(distance)

