import torch
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange
from loguru import logger
import math
import torch.nn.functional as F

def AngleAxisRotatePoint(angleAxis, pt):
    """
    Use angle_axis vector rotate 3D points
    Parameters:
    ------------
    angleAxis : torch.tensor L*3
    pt : torch.tensor L*3
    """
    theta2 = (angleAxis * angleAxis).sum(dim=1)

    mask = (theta2 > 0).float()

    theta = torch.sqrt(theta2 + (1 - mask))

    mask = mask.reshape((mask.shape[0], 1))
    mask = torch.cat([mask, mask, mask], dim=1)

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    thetaInverse = 1.0 / theta

    w0 = angleAxis[:, 0] * thetaInverse
    w1 = angleAxis[:, 1] * thetaInverse
    w2 = angleAxis[:, 2] * thetaInverse

    wCrossPt0 = w1 * pt[:, 2] - w2 * pt[:, 1]
    wCrossPt1 = w2 * pt[:, 0] - w0 * pt[:, 2]
    wCrossPt2 = w0 * pt[:, 1] - w1 * pt[:, 0]

    tmp = (w0 * pt[:, 0] + w1 * pt[:, 1] + w2 * pt[:, 2]) * (1.0 - costheta)

    r0 = pt[:, 0] * costheta + wCrossPt0 * sintheta + w0 * tmp
    r1 = pt[:, 1] * costheta + wCrossPt1 * sintheta + w1 * tmp
    r2 = pt[:, 2] * costheta + wCrossPt2 * sintheta + w2 * tmp

    r0 = r0.reshape((r0.shape[0], 1))
    r1 = r1.reshape((r1.shape[0], 1))
    r2 = r2.reshape((r2.shape[0], 1))

    res1 = torch.cat([r0, r1, r2], dim=1)

    wCrossPt0 = angleAxis[:, 1] * pt[:, 2] - angleAxis[:, 2] * pt[:, 1]
    wCrossPt1 = angleAxis[:, 2] * pt[:, 0] - angleAxis[:, 0] * pt[:, 2]
    wCrossPt2 = angleAxis[:, 0] * pt[:, 1] - angleAxis[:, 1] * pt[:, 0]

    r00 = pt[:, 0] + wCrossPt0
    r01 = pt[:, 1] + wCrossPt1
    r02 = pt[:, 2] + wCrossPt2

    r00 = r00.reshape((r00.shape[0], 1))
    r01 = r01.reshape((r01.shape[0], 1))
    r02 = r02.reshape((r02.shape[0], 1))

    res2 = torch.cat([r00, r01, r02], dim=1)

    return res1 * mask + res2 * (1 - mask)


def coord_normalization(keypoints, h, w, scale=1):
    """
    Normalize keypoints to [-1, 1] for different scales.
    Parameters:
    ---------------
    keypoints: torch.tensor N*2
        coordinates at different images scales
    """
    keypoints = keypoints - scale / 2 + 0.5  # calc down-sampled keypoints positions
    rescale_tensor = torch.tensor([(w - 1) * scale, (h - 1) * scale]).to(keypoints)
    if len(keypoints.shape) == 2:
        rescale_tensor = rescale_tensor[None]
    elif len(keypoints.shape) == 4:
        # grid scenario
        rescale_tensor = rescale_tensor[None, None, None]
    else:
        raise NotImplementedError
    keypoints /= rescale_tensor
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    return keypoints

def sample_feature_from_unfold_featuremap(
    unfold_feature,
    offset=None,
    scale=None,
    mode="OnGrid",
    norm_feature=False,
    return_outof_grid_mask=False,
    patch_feature_size=None,
    verbose=True
):
    """
    Sample feature from unfold feature map(fine level feature map)
    Parameters:
    ---------------
    unfold_feature : torch.tensor L*WW*C
        unfold feature by window size W
    offset : torch.tensor L*2
        target_points - grid_points(feature center points), note that all points are in input image resolution
    scale : input_size / fine_level_size L*2
    mode : str
        choice ['OnGrid', "Offset_Sample]
    norm_feature : bool
        if true: return normalize feature
    return_outof_grid_mask : bool 
        if true: return out of grid mask
    patch_feature_size : int
        size of local patch around keypoints regarding to input image resolution(note: different from sample_feature_from_feature_map,
        it is original image resolution)

    """
    if offset is not None:
        # logger.warning("offset parameter is assigned, sample mode convert to: Offset_Sample")
        mode = "Offset_Sample"
    M, WW, C = unfold_feature.shape
    W = int(math.sqrt(WW))
    if (mode == "OnGrid") and (patch_feature_size is None):
        feat_picked = unfold_feature[:, WW // 2, :]
        out_of_grid_mask = None
    else:
        if mode == "OnGrid":
            offset = torch.zeros((unfold_feature.shape[0], 2), device=unfold_feature.device)

        grid = offset[:, None, None, :]
        if patch_feature_size is not None:
            assert patch_feature_size>0,"invalid patch feature size!"
            local_patch_grid = (
                create_meshgrid(
                    patch_feature_size,
                    patch_feature_size,
                    normalized_coordinates=False,
                    device=unfold_feature.device,
                )
                - patch_feature_size // 2
            )
            grid = grid + local_patch_grid.long()  # L*W*W*2

        offset = offset / (W // 2 * scale)  # normalize offset
        grid = grid / (W // 2 * scale[:, None, None, :]) # normalize grid

        out_of_grid_mask = (offset < -1) | (offset > 1)
        out_of_grid_mask = out_of_grid_mask[:, 0] | out_of_grid_mask[:, 1]

        if verbose:
            if out_of_grid_mask.sum() !=0:
                if out_of_grid_mask.sum() > 200:
                    logger.warning(
                        f"Fine-level Window size is not big enough: w={W}, {out_of_grid_mask.sum()} points locate outside window, total {out_of_grid_mask.shape[0]} points"
                    )
            

        unfold_feature = rearrange(
            unfold_feature, "m (w1 w2) c -> m c w1 w2", w1=W, w2=W
        )

        feat_picked = F.grid_sample(
            unfold_feature, grid, padding_mode="border", align_corners=True
        )
        feat_picked = (
            rearrange(feat_picked, "l c h w -> l h w c").squeeze(-2).squeeze(-2)
        )  # L*c or L*W*W*c

    feat_picked = F.normalize(feat_picked, p=2, dim=-1) if norm_feature else feat_picked

    if return_outof_grid_mask:
        return feat_picked, out_of_grid_mask
    else:
        return feat_picked
