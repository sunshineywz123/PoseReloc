from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import random
import cv2
import matplotlib.cm as cm
import math
from kornia.utils.grid import create_meshgrid
from einops import repeat
from PIL import Image
import h5py
import os.path as osp
from pathlib import Path

matplotlib.use("Agg")

jet = cm.get_cmap("jet")  # "Reds"
jet_colors = jet(np.arange(256))[:, :3]  # color list: normalized to [0,1]


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


def error_colormap(x, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1),
        0,
        1,
    )


def plot_image_pair(imgs, dpi=100, size=6, pad=0.5, horizontal=False):
    n = len(imgs)
    assert n == 2, "number of images must be two"
    if horizontal:
        figsize = (size * n, size) if size is not None else None
        _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    else:
        figsize = (size, size * n) if size is not None else None
        _, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi)

    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_single_image(img, dpi=100, size=6, pad=0.5):
    figsize = (size, size) if size is not None else None
    _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    ax.imshow(img)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for spine in ax.spines.values():  # remove frame
        spine.set_visible(False)
    plt.tight_layout(pad=pad)
    fig = plt.gcf()
    return fig


def plot_keypoints(kpts0, kpts1, color="w", ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps, marker="x")
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps, marker="x")


def plot_keypoints_for_img0(kpts, color="w", ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts[:, 0], kpts[:, 1], c=color, s=ps)


def plot_keypoints_for_img1(kpts, color="w", ps=2):
    ax = plt.gcf().axes
    ax[1].scatter(kpts[:, 0], kpts[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=0.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [
        matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]),
            (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        )
        for i in range(len(kpts0))
    ]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_local_windows(kpts, color="r", lw=1, ax_=0, window_size=9):
    ax = plt.gcf().axes

    patches = []
    for kpt in kpts:
        # patches.append(matplotlib.patches.Rectangle((kpt[0],kpt[1]),window_size,window_size))
        ax[ax_].add_patch(
            matplotlib.patches.Rectangle(
                (kpt[0] - (window_size // 2) - 1, kpt[1] - (window_size // 2) - 1),
                window_size + 2,
                window_size + 2,
                lw=lw,
                color=color,
                fill=False,
            )
        )
    # ax[ax_].add_collection(matplotlib.collections.PathCollection(patches))


def make_matching_plot(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    color,
    text,
    path=None,
    show_keypoints=False,
    fast_viz=False,
    opencv_display=False,
    opencv_title="matches",
    small_text=[],
):

    if fast_viz:
        make_matching_plot_fast(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path,
            show_keypoints,
            10,
            opencv_display,
            opencv_title,
            small_text,
        )
        return

    plot_image_pair([image0, image1])  # will create a new figure
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color="k", ps=4)
        plot_keypoints(kpts0, kpts1, color="w", ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = "k" if image0[:100, :150].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    txt_color = "k" if image0[-100:, :150].mean() > 200 else "w"
    fig.text(
        0.01,
        0.01,
        "\n".join(small_text),
        transform=fig.axes[0].transAxes,
        fontsize=5,
        va="bottom",
        ha="left",
        color=txt_color,
    )
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        # TODO: Would it leads to any issue without current figure opened?
        return fig


def make_matching_plot_show_more(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    color,
    text,
    path=None,
    show_keypoints=False,
    draw_matches=True,
    fast_viz=False,
    opencv_display=False,
    opencv_title="matches",
    small_text=[],
    keypoints_more_0_a=None,
    keypoints_more_0_b=None,
    keypoints_more_1_a=None,
    draw_local_window=None,
    window_size=9,
    mkpts_mask=None,
):

    if fast_viz:
        make_matching_plot_fast(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path,
            show_keypoints,
            10,
            opencv_display,
            opencv_title,
            small_text,
        )
        return

    plot_image_pair([image0, image1], dpi=600)  # will create a new figure

    blue_points_size = 0.3  # means coarse full matches
    yellow_points_size = 0.1  # means detector's output keypoints
    green_points_size = 0.1  # means final matches

    print(
        f"coarse: {len(keypoints_more_0_a)}, spp keypoints:{len(keypoints_more_0_b)}, total matches:{len(mkpts0)}"
    )
    if keypoints_more_0_a is not None:
        plot_keypoints_for_img0(keypoints_more_0_a, color="blue", ps=blue_points_size)
    if keypoints_more_0_b is not None:
        plot_keypoints_for_img0(
            keypoints_more_0_b, color="yellow", ps=yellow_points_size
        )
    if keypoints_more_1_a is not None:
        plot_keypoints_for_img1(keypoints_more_1_a, color="blue", ps=blue_points_size)
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color="green", ps=green_points_size)

    if draw_matches:
        mkpts0_ = mkpts0[mkpts_mask > 0] if mkpts_mask is not None else mkpts0
        mkpts1_ = mkpts1[mkpts_mask > 0] if mkpts_mask is not None else mkpts1
        n = len(mkpts0_)
        n_matches = 10000
        if n < n_matches:
            kpts_indices = np.arange(n)
        else:
            kpts_indices = random.sample(range(0, n), n_matches)
        mkpts0_, mkpts1_ = map(lambda x: x[kpts_indices], [mkpts0_, mkpts1_])
        colors = []
        for i in range(len(mkpts0_)):
            colors.append(color)
        plot_matches(mkpts0_, mkpts1_, colors, lw=0.1, ps=0)

    if draw_local_window is not None:
        for type in draw_local_window:
            if "0_a" in type:
                assert keypoints_more_0_a is not None
                plot_local_windows(
                    keypoints_more_0_a,
                    color="r",
                    lw=0.1,
                    ax_=0,
                    window_size=window_size,
                )
            if "0_b" in type:
                assert keypoints_more_0_b is not None
                plot_local_windows(
                    keypoints_more_0_b,
                    color="r",
                    lw=0.1,
                    ax_=0,
                    window_size=window_size,
                )
            if "1_a" in type:
                assert keypoints_more_1_a is not None
                plot_local_windows(
                    keypoints_more_1_a,
                    color="r",
                    lw=0.1,
                    ax_=1,
                    window_size=window_size,
                )

    fig = plt.gcf()
    txt_color = "k" if image0[:100, :150].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    txt_color = "k" if image0[-100:, :150].mean() > 200 else "w"
    fig.text(
        0.01,
        0.01,
        "\n".join(small_text),
        transform=fig.axes[0].transAxes,
        fontsize=5,
        va="bottom",
        ha="left",
        color=txt_color,
    )
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        # plt.close()
    # TODO: Would it leads to any issue without current figure opened?
    return fig


def make_matching_plot_fast(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    color,
    text,
    path=None,
    show_keypoints=False,
    margin=10,
    opencv_display=False,
    opencv_title="",
    small_text=[],
):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin :] = image1
    out = np.stack([out] * 3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(
            out,
            (x0, y0),
            (x1 + margin + W0, y1),
            color=c,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640.0, 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def reproj(K, pose, pts_3d):
    """ 
    Reproj 3d points to 2d points 
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K

    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose

    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    depth = reproj_points[2]  # [N]
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T

    return reproj_points, depth  # [n, 2]


@torch.no_grad()
def draw_reprojection_pair(
    data, visual_color_type="conf", visual_gt=False, visual_heatmap=False
):
    figures = {"evaluation": []}
    # figures = {"evaluation": [], "heatmap":[]}

    if visual_gt:
        raise NotImplementedError
        # For gt debug
        # NOTE: only available for batch size = 1
        query_image = (data["query_image"].cpu().numpy() * 255).round().astype(np.int32)
        if query_image.shape[0] != 1:
            logger.warning("Not implement visual gt for batch size != 0")
            return

        mkpts_3d = (
            data["keypoints3d"][0, data["mkpts3D_idx_gt"][0]].cpu().numpy()
        )  # GT mkpts3D
        mkpts_query = data["mkpts2D_gt"][0].cpu().numpy()
        m_bids = np.zeros((mkpts_3d.shape[0],))
        query_K = data["query_intrinsic"].cpu().numpy()
        query_pose_gt = data["query_pose_gt"].cpu().numpy()  # B*4*4
        m_conf = np.zeros((mkpts_3d.shape[0],))

    else:
        m_bids = data["m_bids"].cpu().numpy()
        query_image = (data["query_image"].cpu().numpy() * 255).round().astype(np.int32)
        mkpts_3d = data["mkpts_3d_db"].cpu().numpy()
        mkpts_query_c = data["mkpts_query_c"].cpu().numpy()
        mkpts_query = data["mkpts_query_f"].cpu().numpy()
        query_K = data["query_intrinsic"].cpu().numpy()
        query_pose_gt = data["query_pose_gt"].cpu().numpy()  # B*4*4
        m_conf = data["mconf"].cpu().numpy()
        gt_mask = data['gt_mask'].cpu().numpy()

    R_errs = data["R_errs"] if "R_errs" in data else None
    t_errs = data["t_errs"] if "t_errs" in data else None
    inliers = data["inliers"] if "inliers" in data else None

    R_errs_c = data["R_errs_c"] if "R_errs_c" in data else None
    t_errs_c = data["t_errs_c"] if "t_errs_c" in data else None
    inliers_c = data["inliers_c"] if "inliers_c" in data else None

    R_errs_gt = data["R_errs_gt"] if "R_errs_gt" in data else None
    t_errs_gt = data["t_errs_gt"] if "t_errs_gt" in data else None
    inliers_gt = data["inliers_gt"] if "inliers_gt" in data else None

    if visual_heatmap:
        feat3d = data["desc3d_db_selected"]
        feat2d_unfold_f = data["query_feat_f_unfolded"]
        M, WW, C = feat2d_unfold_f.shape
        W = int(math.sqrt(WW))
        sim_matrix = torch.linalg.norm(feat3d - feat2d_unfold_f, dim=-1)
        sim_matrix = sim_matrix.cpu().numpy()

        sim_matrix_max = np.max(sim_matrix, axis=-1, keepdims=True)
        sim_matrix_min = np.min(sim_matrix, axis=-1, keepdims=True)
        sim_matrix_normalized = (sim_matrix - sim_matrix_min) / (
            sim_matrix_max - sim_matrix_min + 1e-4
        )
        sim_matrix_normalized = np.reshape(sim_matrix_normalized, (-1, W, W))

        grid = (
            (create_meshgrid(W, W, normalized_coordinates=False,) - W // 2)
            .cpu()
            .numpy()
        )

    for bs in range(data["query_image"].size(0)):
        mask = m_bids == bs

        mkpts3d_reprojed, depth = reproj(query_K[bs], query_pose_gt[bs], mkpts_3d[mask])
        mkpts_query_masked = mkpts_query[mask]

        if "query_image_scale" in data:
            mkpts3d_reprojed = (
                mkpts3d_reprojed / data["query_image_scale"][bs].cpu().numpy()[[1, 0]]
            )
            mkpts_query_masked = (
                mkpts_query_masked / data["query_image_scale"][bs].cpu().numpy()[[1, 0]]
            )

        text = [
            f"Num of matches: {mkpts3d_reprojed.shape[0]}",
        ]

        if R_errs is not None:
            text += [f"R_err: {R_errs[bs]}"]
        if t_errs is not None:
            text += [f"t_err: {t_errs[bs]}"]
        if inliers is not None:
            text += [
                f"Num of inliers: {inliers[bs].shape[0] if not isinstance(inliers[bs], list) else len(inliers[bs])}"
            ]

        if R_errs_c is not None:
            text += [f"R_err_coarse: {R_errs_c[bs]}"]
        if t_errs_c is not None:
            text += [f"t_err_coarse: {t_errs_c[bs]}"]
        if inliers_c is not None:
            text += [
                f"Num of inliers coarse: {inliers_c[bs].shape[0] if not isinstance(inliers[bs], list) else len(inliers_c[bs])}"
            ]

        if R_errs_gt is not None:
            text += [f"R_err_gt: {R_errs_gt[bs]}"]
        if t_errs_gt is not None:
            text += [f"t_err_gt: {t_errs_gt[bs]}"]
        if inliers_gt is not None:
            text += [
                f"Num of inliers gt: {inliers_gt[bs].shape[0] if not isinstance(inliers[bs], list) else len(inliers_gt[bs])}"
            ]

        # Clip reprojected keypoints
        # FIXME: bug exists here! a_max need to be original image size
        mkpts3d_reprojed[:, 0] = np.clip(
            mkpts3d_reprojed[:, 0], a_min=0, a_max=data["query_image"].shape[-1] - 1
        )  # x
        mkpts3d_reprojed[:, 1] = np.clip(
            mkpts3d_reprojed[:, 1], a_min=0, a_max=data["query_image"].shape[-2] - 1
        )  # y

        if visual_color_type == "conf":
            if mkpts3d_reprojed.shape[0] != 0:
                m_conf_max = np.max(m_conf[mask])
                m_conf_min = np.min(m_conf[mask])
                m_conf_normalized = (m_conf[mask] - m_conf_min) / (
                    m_conf_max - m_conf_min + 1e-4
                )
                color = jet(m_conf_normalized)

                text += [
                    f"Max conf: {m_conf_max}",
                    f"Min conf: {m_conf_min}",
                ]
            else:
                color = np.array([])
        elif visual_color_type == "distance_error":
            color_thr = 5
            reprojection_dictance = np.linalg.norm(
                mkpts3d_reprojed - mkpts_query_masked, axis=-1
            )
            color = np.clip(reprojection_dictance / (color_thr), 0, 1)
            color = error_colormap(1 - color, alpha=0.5)
        elif visual_color_type == "depth":
            if depth.shape[0] != 0:
                depth_max = np.max(depth)
                depth_min = np.min(depth)
                depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-4)
                color = jet(depth_normalized)
            else:
                color = np.array([])

        elif visual_color_type == "true_or_false":
            # NOTE: only available for train to visual whether coarse match is correct
            pass
        else:
            raise NotImplementedError

        figure = make_matching_plot(
            query_image[bs][0],
            query_image[bs][0],
            mkpts_query_masked,
            mkpts3d_reprojed,
            mkpts_query_masked,
            mkpts3d_reprojed,
            color=color,
            text=text,
        )
        figures["evaluation"].append(figure)

        if visual_heatmap:
            sim_matrix_normalized_masked = sim_matrix_normalized[~gt_mask][mask]
            coordinate_grid = (mkpts_query_c[:, None, None, :] + grid).astype(
                np.int
            )  # L*W*W*2
            h, w = query_image[bs][0].shape[:2]

            response_map = np.zeros((h, w))
            response_map[
                coordinate_grid[..., 1], coordinate_grid[..., 0]
            ] = sim_matrix_normalized_masked  # h*w
            response_map = np.uint8(255 * response_map)
            colored_response_map = jet_colors[response_map]  # h*w*3
            img_blend_with_heatmap = blend_img_heatmap(
                query_image[bs][0], colored_response_map, alpha=0.5
            )
            fig = plot_single_image(img_blend_with_heatmap, dpi=300, size=6)
            figures["heatmap"].append(fig)

        return figures

def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))

def plot_response_map(keypoint_h5_path, feature_patch_h5_path, match_h5_path, pairs_path, save_dir=""):
    if not osp.exists(feature_patch_h5_path):
        logger.warning(f"{feature_patch_h5_path} not exists!")
        return

    # Load keypoints:
    keypoint_file = h5py.File(str(keypoint_h5_path), 'r')
    # Load matches:
    match_file = h5py.File(str(match_h5_path), 'r')
    # Load features:
    feature_file = h5py.File(str(feature_patch_h5_path), 'r')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split(' ') for p in f.read().split('\n')]
    
    for id, (name0, name1) in enumerate(pairs):
        pair = names_to_pair(name0, name1)
        matches = match_file[pair]['matches'].__array__()

        keypoints0 = keypoint_file[name0]['keypoints'].__array__()
        keypoints1 = keypoint_file[name1]['keypoints'].__array__()

        features_id0 = feature_file[name0]["keypoint_ids"].__array__()
        features_id1 = feature_file[name1]["keypoint_ids"].__array__()
        features_patch0 = feature_file[name0]["patches"].__array__()
        features_patch1 = feature_file[name1]["patches"].__array__()

        mkpts0, mkpts1 = keypoints0[matches[:, 0]], keypoints1[matches[:, 1]]
        m_feat_patches0, m_feat_patches1 = [], []
        for kpt_id0, kpt_id1 in matches:
            # Get features:
            patch_idx0 = np.argwhere(features_id0 == kpt_id0)[0,0]
            patch_idx1 = np.argwhere(features_id1 == kpt_id1)[0,0]
            feature_patch0, feature_patch1 = features_patch0[patch_idx0], features_patch1[patch_idx1]
            m_feat_patches0.append(feature_patch0)
            m_feat_patches1.append(feature_patch1)
        m_feat_patches0, m_feat_patches1 = np.stack(m_feat_patches0), np.stack(m_feat_patches1)

        m_feature_query = m_feat_patches0[:, [4], [4], :][:, None]
        sim_matrix = np.linalg.norm(m_feat_patches1 - m_feature_query, axis=-1)
        sim_matrix_max = np.max(sim_matrix, axis=-1, keepdims=True)
        sim_matrix_min = np.min(sim_matrix, axis=-1, keepdims=True)
        sim_matrix_normalized = (sim_matrix - sim_matrix_min) / (
            sim_matrix_max - sim_matrix_min + 1e-4
        )
        # image0 = np.array(Image.open(name0))
        image1 = np.array(Image.open(name1))

        W = 8
        grid = (
            (create_meshgrid(W, W, normalized_coordinates=False,) - W // 2)
            .cpu()
            .numpy()
        )

        coordinate_grid = (mkpts1[:, None, None, :] + grid).astype(
            np.int
        )  # L*W*W*2
        h, w = image1.shape[:2]

        response_map = np.zeros((h, w))
        response_map[
            coordinate_grid[..., 1], coordinate_grid[..., 0]
        ] = sim_matrix_normalized  # h*w
        response_map = np.uint8(255 * response_map)
        colored_response_map = jet_colors[response_map]  # h*w*3
        img_blend_with_heatmap = blend_img_heatmap(
            image1, colored_response_map, alpha=0.5
        )
        fig = plot_single_image(img_blend_with_heatmap, dpi=300, size=6)
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(osp.join(save_dir, f"pixsfm_{osp.splitext(osp.basename(name0))[0]}-{osp.splitext(osp.basename(name1))[0]}.png"), bbox_inches="tight", pad_inches=0)

        a = 1


    # Plot local response map

