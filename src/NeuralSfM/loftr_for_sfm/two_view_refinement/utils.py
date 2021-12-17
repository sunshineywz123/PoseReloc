from kornia.utils.grid import create_meshgrid
from matplotlib import pyplot as plt
import imageio
import io
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import math
from einops.einops import rearrange
from loguru import logger
import matplotlib.cm as cm
import cv2
from einops import repeat
import os
import os.path as osp

jet = cm.get_cmap("jet")  # "Reds"
jet_colors = jet(np.arange(256))[:, :3]  # color list: normalized to [0,1]


def find_all_character(name, aim_character):
    for i, character in enumerate(name):
        if character == aim_character:
            yield i


def blend_img_heatmap(img, heatmap, alpha=0.5):  # (H, W, *)  # (H, W, 3)
    h, w = heatmap.shape[:2]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2:
        img = repeat(img, "h w -> h w c", c=3)
    img = np.uint8(img)
    heatmap = np.uint8(heatmap * 255)
    blended = np.asarray(
        Image.blend(Image.fromarray(img), Image.fromarray(heatmap), alpha)
    )
    return blended


def add_direction_to_heat_map_image(
    img_blend_heatmap,
    reprojection_kpts_list,
    matched_kpts=None,
    centerpoints=None,
    dpi=100,
    save_path=None,
    save_format=None,
    **kwargs,
):
    """
    add optimize directions to images blend with heatmap, and save as png use matplotlib
    Parameters:
    --------------
    img_blend_heatmap : numpy.array h*w*3
    reprojection_kpts_list : List[numpy.array L*2] n>=2

    """
    h, w, c = img_blend_heatmap.shape
    # fig, ax = plt.subplots(1, 1, figsize=(w / float(dpi), h / float(dpi)), dpi=dpi)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20), dpi=dpi)

    ax.imshow(img_blend_heatmap, alpha=0.6)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for spine in ax.spines.values():  # remove frame
        spine.set_visible(False)
    plt.tight_layout(pad=0.5)

    # build tragectory sequence
    tragectory_sequence = zip(reprojection_kpts_list[:-1], reprojection_kpts_list[1:])
    color_grid = 1 / len(reprojection_kpts_list[:-1])

    if matched_kpts is not None:
        ax.scatter(
            matched_kpts[:, 0], matched_kpts[:, 1], color="white", s=2, marker="x"
        )
    if centerpoints is not None:
        ax.scatter(
            centerpoints[:, 0], centerpoints[:, 1], color="black", s=2, marker="x"
        )

    for i, [now_coord, next_coord] in enumerate(tragectory_sequence):
        direction = next_coord - now_coord
        ax.quiver(
            now_coord[:, 0],
            now_coord[:, 1],
            direction[:, 0],
            direction[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=1,
            headlength=1,
            # color=jet_colors[i * color_grid],
            color=cm.jet(i * color_grid, alpha=0.4),
        )
        # if save_path is not None:
        #     plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    if save_path is not None:
        if save_format is None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        else:
            save_path = osp.splitext(save_path)[0] + save_format
            plt.savefig(save_path)
    return fig


def add_reprojection_points_to_heat_map_image(
    img_blend_heatmap,
    reprojection_kpts_list,
    matched_kpts=None,
    centerpoints=None,
    dpi=100,
    save_path=None,
    save_format="pdf",
    big_mark_index=None,
    **kwargs,
):
    """
    add optimize directions to images blend with heatmap, and save as png use matplotlib
    Parameters:
    --------------
    img_blend_heatmap : numpy.array h*w*3
    reprojection_kpts_list : List[List[numpy.array L*2] n] k>=2
    matched_kpts : numpy.array L*2
        matched kpts by LoFTR fine
    centerpoints : numpy.array L*2
        fine local grids' center points coordinate

    """
    h, w, c = img_blend_heatmap.shape
    # fig, ax = plt.subplots(1, 1, figsize=(w / float(dpi), h / float(dpi)), dpi=dpi)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20), dpi=dpi)

    ax.imshow(img_blend_heatmap, alpha=0.6)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for spine in ax.spines.values():  # remove frame
        spine.set_visible(False)
    plt.tight_layout(pad=0.5)

    if matched_kpts is not None:
        ax.scatter(
            matched_kpts[:, 0], matched_kpts[:, 1], color="green", s=1, marker="x"
        )
    if centerpoints is not None:
        ax.scatter(
            centerpoints[:, 0], centerpoints[:, 1], color="black", s=1, marker="x"
        )

    # TODO: move these paras to outside
    draw_all_points = False  # whether draw all reprojection points, if not: only draw begin and end points
    draw_mode = "all"  # choice: ['all', 'only_pose', 'only_depth', 'only_BA]
    draw_optim_step_dir = (
        False  # from each optim sub-part begin points to sub-part end points
    )
    draw_global_optim_dir = False  # from optim begin points to optim end points

    if draw_all_points:
        outer_color_grid = 255 // len(reprojection_kpts_list)
        outer_color_grid = 1 if outer_color_grid == 0 else outer_color_grid
        for i, reprojection_kpts_for_each_optimization in enumerate(
            reprojection_kpts_list
        ):
            # NOTE: only available for refine_type == 'depth_pose', otherwise inverse "only_pose" and "only_depth"
            if draw_mode == "all":
                pass
            elif draw_mode == "only_pose":
                if i % 2 == 0:
                    continue
            elif draw_mode == "only_depth":
                if i % 2 != 0 or i == len(reprojection_kpts_list) - 1:
                    continue
            elif draw_mode == "only_BA":
                if i != len(reprojection_kpts_list) - 1:
                    continue

            color_grid = 1 / len(reprojection_kpts_for_each_optimization)
            for j, reprojection_kpts in enumerate(
                reprojection_kpts_for_each_optimization
            ):
                if j != len(reprojection_kpts_for_each_optimization) - 1:
                    ax.scatter(
                        reprojection_kpts[:, 0],
                        reprojection_kpts[:, 1],
                        # color=jet_colors[i * color_grid],
                        color=cm.jet(j * color_grid, alpha=0.4),
                        s=0.5,
                        marker="o",
                    )
                else:
                    # the last point
                    ax.scatter(
                        reprojection_kpts[:, 0],
                        reprojection_kpts[:, 1],
                        # color=jet_colors[i * color_grid],
                        color=cm.jet(j * color_grid, alpha=0.8),
                        s=1,
                        marker="x",
                    )

            if draw_optim_step_dir:
                # Draw every optimization direction from begin to end
                begin = reprojection_kpts_for_each_optimization[0]
                end = reprojection_kpts_for_each_optimization[-1]
                direction = (end - begin) * 2
                direction = 4 * (
                    direction
                    / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-4)
                )  # normalize direction and rescale for  visualize
                ax.quiver(
                    begin[:, 0],
                    begin[:, 1],
                    direction[:, 0],
                    direction[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    headwidth=1,
                    headlength=1,
                    # color=jet_colors[i * color_grid],
                    color=cm.jet(
                        i * outer_color_grid, alpha=0.5
                    ),  # color denote optimization process: blue operation[0] red:
                )
    else:
        # only draw optimization start points and optimization end points
        # start points
        ax.scatter(
            reprojection_kpts_list[0][0][:, 0],
            reprojection_kpts_list[0][0][:, 1],
            # color=jet_colors[i * color_grid],
            color=cm.jet(0.0, alpha=0.8),
            s=1,
            marker="o",
        )
        # end points
        ax.scatter(
            reprojection_kpts_list[-1][-1][:, 0],
            reprojection_kpts_list[-1][-1][:, 1],
            # color=jet_colors[i * color_grid],
            color=cm.jet(1.0, alpha=0.8),
            s=1,
            marker="o",
        )

    if draw_global_optim_dir:
        # Draw arrow from optimization begin to optimization end
        begin = reprojection_kpts_list[0][0]
        end = reprojection_kpts_list[-1][-1]
        direction = (end - begin) * 2
        direction = 6 * (
            direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-4)
        )  # normalize direction and rescale for  visualize
        ax.quiver(
            begin[:, 0],
            begin[:, 1],
            direction[:, 0],
            direction[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=1,
            headlength=1,
            # color=jet_colors[i * color_grid],
            color=(1, 1, 1, 0.7),
        )

    # mark wanted grid
    if big_mark_index is not None:
        ax.scatter(
            centerpoints[big_mark_index][:, 0],
            centerpoints[big_mark_index][:, 1],
            color="red",
            s=40,
            marker="x",
        )

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        if save_format is not None:
            save_path = osp.splitext(save_path)[0] + "." + save_format
            plt.savefig(save_path)
    return fig


def draw_local_grid_rpjpts_single(
    local_heatmap,
    reprojected_points_offset,
    matched_kpt_offset=None,
    save_path=None,
    make_gif=True,
):
    """
    draw single local grid and reprojected_points
    Parameters:
    --------------
    local_heatmap : numpy.array w*w*3
    reprojected_points_offset : List[numpy.array [2]] n
        reprojected_points_offset is relative to local heatmap center point
    matches_kpt_offset : numpy.array [2]
    """

    h, w = local_heatmap.shape[:2]
    fig, ax = plt.subplots(1, 1)
    ax.imshow(local_heatmap)
    # reprojected_points = np.concatenate(reprojected_points, axis=0)

    if matched_kpt_offset is not None:
        ax.scatter(
            matched_kpt_offset[0] + w // 2,
            matched_kpt_offset[1] + h // 2,
            color="green",
            marker="x",
        )

    img_list = [] if make_gif else None
    for i, rpj_pts_each_optim in enumerate(reprojected_points_offset):

        color_grid = 1 / len(rpj_pts_each_optim)
        for j, rpj_pts in enumerate(rpj_pts_each_optim):
            if max(abs(rpj_pts)) > 200:
                logger.warning(f"keypoint very far from local grid center, max: {max(abs(rpj_pts))}")
                # continue
            if j != len(rpj_pts_each_optim) - 1:
                ax.scatter(
                    rpj_pts[0] + w // 2,
                    rpj_pts[1] + h // 2,
                    color=cm.jet(j * color_grid, alpha=0.4),
                )
            else:
                ax.scatter(
                    rpj_pts[0] + w // 2,
                    rpj_pts[1] + h // 2,
                    color=cm.jet(j * color_grid, alpha=0.8),
                    marker="x",
                )

            if img_list is not None:
                io_buf = io.BytesIO()
                fig.savefig(io_buf, format="raw")
                io_buf.seek(0)
                img = np.reshape(
                    np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                    newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
                )
                io_buf.close()
                img_list += [img]

            # if save_path is not None:
            #     plt.savefig(save_path)

    if make_gif is not None:
        gif_save_path = osp.splitext(save_path)[0] + ".gif"
        imageio.mimsave(gif_save_path, img_list, fps=20)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def draw_heatmap_of_local_patch(
    image_path,
    query_features,
    feature_patch,
    centerpoints,
    reprojected_kpts_list=None,
    matched_kpts=None,
    visual_type="correlation",
    save_dir=None,
    scale=1,
):
    """
    Get query features' responce on feature_patch, draw on image and save
    Parameters:
    ------------
    query_features: torch.tensor m*c
        sparse feature in S2D
    feature_patch: torch.tensor m*ww*c
        local feature patch
    centerpoints: torch.tensor m*2
        feature_patch's center point location in image coordinate
    reprojected_kpts_list: List[torch.tensor L*2]
        list of reprojected keypoints for drawing optimization trajectory
    matched_kpts: torch.tensor m*2
        keypoints matched by loftr
    visual_type: str
        choice: ['correlation', 'distance']
    scale:
        scale = image_scale/feature_scale
    
    """
    # Get heatmap
    M, WW, C = feature_patch.shape
    W = int(math.sqrt(WW))
    if len(query_features.shape) != 2:
        # patch alignment scenario
        patch_w = query_features.shape[1]
        query_features = query_features[:,patch_w//2, patch_w//2,:]

    if visual_type == "correlation":
        sim_matrix = torch.einsum("mc,mrc->mr", query_features, feature_patch)
    elif visual_type == "distance":
        sim_matrix = torch.linalg.norm(
            query_features.unsqueeze(1) - feature_patch, dim=-1
        )
    else:
        raise NotImplementedError
    softmax_temp = 1.0 / C ** 0.5
    # heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)  # M*WW
    heatmap = softmax_temp * sim_matrix

    # normalize heatmap to [0,1]
    heatmap_maxim, _ = torch.max(heatmap, dim=-1, keepdim=True)
    heatmap_min, _ = torch.min(heatmap, dim=-1, keepdim=True)
    heatmap_normalized = (heatmap - heatmap_min) / (heatmap_maxim - heatmap_min)
    heatmap_normalized = heatmap_normalized.view(-1, W, W)  # M*W*W

    # W = W * scale

    # Get every point's coordinate in each grid in image resolution
    grid = (
        create_meshgrid(W, W, normalized_coordinates=False, device=heatmap.device)
        - W // 2
    )
    coordinate_grid = (centerpoints[:, None, None, :] + grid).long()  # L*W*W*2

    # Load image
    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]

    # Aggregate local grid heat map and get response map
    response_map = torch.zeros((h, w), device=heatmap.device)
    response_map[
        coordinate_grid[..., 1], coordinate_grid[..., 0]
    ] = heatmap_normalized  # h*w

    # set color to heatmap
    response_map = np.uint8(255 * response_map.cpu())
    colored_response_map = jet_colors[response_map]  # h*w*3

    img_blend_with_heatmap = blend_img_heatmap(image, colored_response_map, alpha=0.5)

    # keypoints = keypoints.cpu().numpy()
    # for i, keypoint in enumerate(keypoints):
    #     img_blend_with_heatmap = cv2.arrowedLine(img_blend_with_heatmap, tuple(keypoint), tuple(keypoint+4), color=(0,0,0))

    # save image blend with heatmap as png format
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(
        "" if save_dir is None else save_dir,
        "_".join(
            [
                "heat_map",
                visual_type,
                osp.splitext(osp.basename(image_path))[0] + ".png",
            ]
        ),
    )
    Image.fromarray(img_blend_with_heatmap).save(save_path)

    want_index = 40
    if reprojected_kpts_list is not None:
        # non-empty check
        reprojected_kpts_list = [
            reprojected_kpts
            for reprojected_kpts in reprojected_kpts_list
            if len(reprojected_kpts) != 0
        ]
        if save_dir is not None:
            save_dir = osp.join(save_dir, "heatmap_with_direction")
            os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(
            "" if save_dir is None else save_dir,
            "_".join(
                [
                    "heat_map_with_direction",
                    visual_type,
                    osp.splitext(osp.basename(image_path))[0] + ".png",
                ]
            ),
        )
        dpi = 300

        # Rescale reprojected kpts relative center points of feature grid
        rescaled_reprojected_kpts_list = [
            [
                ((kpts - centerpoints) / scale + centerpoints).cpu().numpy()
                for kpts in reprojected_kpts
            ]
            for reprojected_kpts in reprojected_kpts_list
        ]

        add_direction = False
        add_item_to_image = (
            add_direction_to_heat_map_image
            if add_direction
            else add_reprojection_points_to_heat_map_image
        )
        fig = add_item_to_image(
            img_blend_with_heatmap,
            rescaled_reprojected_kpts_list,
            dpi=dpi,
            matched_kpts=((matched_kpts - centerpoints) / scale + centerpoints)
            .cpu()
            .numpy()
            if matched_kpts is not None
            else None,
            centerpoints=centerpoints.cpu().numpy(),
            save_path=save_path,
            big_mark_index=np.array([want_index]),
        )
        plt.close(fig)

        # draw single local grid heatmap and reprojected points
        single_local_grid_heatmap = np.uint8(255 * heatmap_normalized[want_index].cpu())
        colored_single_local_grid = jet_colors[single_local_grid_heatmap]
        reprojected_kpt = [
            [
                ((kpts[want_index] - centerpoints[want_index]) / scale).cpu().numpy()
                for kpts in reprojected_kpts
            ]
            for reprojected_kpts in reprojected_kpts_list
        ]
        draw_local_grid_rpjpts_single(
            colored_single_local_grid,
            reprojected_kpt,
            matched_kpt_offset=(
                (matched_kpts[want_index] - centerpoints[want_index]) / scale
            )
            .cpu()
            .numpy(),
            save_path=osp.splitext(save_path)[0] + "_want_index"
            # + f"_want_index_{want_index}"
            + osp.splitext(save_path)[1],
        )
        pass
    pass


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


def sample_feature_from_featuremap(
    feature_map, kpts, imghw, norm_feature=False, patch_feature_size=None
):
    """
    Sample feature from whold feature map
    Parameters:
    -------------
    feature_map : torch.tensor C*H*W
    kpts : torch.tensor L*2
    imghw : torch.tensor 2
        h=imghw[0], w=imghw[1]
    norm_feature : bool
        if true: return normalize feature
    patch_feature_size : int
        size of local patch around keypoints regarding to original image resolution
    """
    c, h, w = feature_map.shape

    # grid = kpts[:, None, None, :]  # L*1*1*2 TODO: 1*L*1*2
    grid = kpts[None, :, None, :] # 1*L*1*2
    if patch_feature_size is not None:
        assert patch_feature_size>0,"invalid patch feature size!"
        # Get every point's coordinate in each grid in image resolution
        local_patch_grid = (
            create_meshgrid(
                patch_feature_size,
                patch_feature_size,
                normalized_coordinates=False,
                device=feature_map.device,
            )
            - patch_feature_size // 2
        )
        grid = grid.unsqueeze(-2) # 1*L*1*1*2
        grid = grid + local_patch_grid.long().unsqueeze(0)  # 1*L*W*W*2
        grid = rearrange(grid, "n l h w c -> n l (h w) c") # 1*L*WW*2

    # FIXME: problem here: local window is also rescaled!
    grid_n = coord_normalization(grid, imghw[0], imghw[1])

    feature = F.grid_sample(
        feature_map.unsqueeze(0),  # 1*C*H*W
        grid_n,
        mode="bilinear",
        align_corners=True,
        padding_mode="reflection",
    )  # 1*C*L*WW or 1*C*L*1
    feature = (
        rearrange(feature, "l c h w -> l h w c").squeeze(0)
    )  # L*WW*C or L*1*C

    if patch_feature_size is not None:
        feature = rearrange(feature, "l (h w) c -> l h w c", h=patch_feature_size) # L*W*W*C
    else:
        feature = feature.squeeze(-2) # L*C

    return F.normalize(feature, p=2, dim=-1) if norm_feature else feature


def sample_feature_from_unfold_featuremap(
    unfold_feature,
    offset=None,
    scale=None,
    mode="OnGrid",
    norm_feature=False,
    return_outof_grid_mask=False,
    patch_feature_size=None
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
        grid = grid / (W // 2 * scale) # normalize grid

        out_of_grid_mask = (offset < -1) | (offset > 1)
        out_of_grid_mask = out_of_grid_mask[:, 0] | out_of_grid_mask[:, 1]

        if out_of_grid_mask.sum() !=0:
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
