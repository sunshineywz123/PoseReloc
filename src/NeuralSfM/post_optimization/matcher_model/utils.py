import numpy as np
from kornia.utils.grid import create_meshgrid
import torch
import numpy as np
import torch.nn.functional as F
import math
from einops.einops import rearrange
from loguru import logger


def sample_feature_from_unfold_featuremap(
    unfold_feature,
    offset=None,
    scale=2,
    mode="OnGrid",
    norm_feature=False,
    return_outof_grid_mask=False,
    patch_feature_size=None,
):
    """
    Sample feature from unfold feature map(fine level feature map)
    Parameters:
    ---------------
    unfold_feature : torch.tensor L*WW*C
        unfold feature by window size W
    offset : torch.tensor L*2
        target_points - grid_points(feature center points), note that all points are in input image resolution
    scale : input_size / fine_level_size
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
            offset = torch.zeros(
                (unfold_feature.shape[0], 2), device=unfold_feature.device
            )

        grid = offset[:, None, None, :]
        if patch_feature_size is not None:
            assert patch_feature_size > 0, "invalid patch feature size!"
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
        grid = grid / (W // 2 * scale)  # normalize grid

        out_of_grid_mask = (offset < -1) | (offset > 1)
        out_of_grid_mask = out_of_grid_mask[:, 0] | out_of_grid_mask[:, 1]

        if out_of_grid_mask.sum() != 0:
            logger.warning(
                f"Fine-level Window size is not big enough: w={W}, {out_of_grid_mask.sum()} points locate outside window, total {out_of_grid_mask.shape[0]} points"
            )

        unfold_feature = rearrange(
            unfold_feature, "m (w1 w2) c -> m c w1 w2", w1=W, w2=W
        )

        feat_picked = F.grid_sample(
            unfold_feature, grid.float(), padding_mode="border", align_corners=True
        )
        feat_picked = (
            rearrange(feat_picked, "l c h w -> l h w c").squeeze(-2).squeeze(-2)
        )  # L*c or L*W*W*c

    feat_picked = F.normalize(feat_picked, p=2, dim=-1) if norm_feature else feat_picked

    if return_outof_grid_mask:
        return feat_picked, out_of_grid_mask
    else:
        return feat_picked


def buildDataPair(data0, data1):
    # data0: dict, data1: dict
    data = {}
    for i, data_part in enumerate([data0, data1]):
        for key, value in data_part.items():
            data[key + str(i)] = value
    assert len(data) % 2 == 0, "build data pair error! please check built data pair!"
    data["pair_names"] = (data["img_path0"], data["img_path1"])
    return data