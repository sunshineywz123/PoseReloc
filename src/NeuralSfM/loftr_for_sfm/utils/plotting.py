from matplotlib.pyplot import figure
import numpy as np
from src.utils.utils import make_matching_plot, error_colormap, make_matching_plot_show_more


def draw_sinlge_evaluation(data, bs):
    mask = data['m_bids'] == bs
    thr = 5e-4

    image0 = (data['image0'][bs][0].cpu().numpy() * 255).round().astype(np.int32)
    image1 = (data['image1'][bs][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][mask].cpu().numpy()
    
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][bs].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][bs].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][mask].cpu().numpy()
    correct_mask = epi_errs < thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][bs].sum().cpu())
    recall = n_correct / n_gt_matches

    # Display matching info
    color = np.clip(epi_errs / (thr*2), 0, 1)
    color = error_colormap(1 - color, alpha=0.5)  # 0.1 is too vague
    # TODO: Precision & Recall for Outlier Rejection
    text = [
        f'Matches {len(kpts0)}',
        f'Precision({thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    if "coarse_full_match_points0" in data:
        coarse_full_mask=data['coarse_full_match_b_ids']==bs
        coarse_full_match_points0 = data['coarse_full_match_points0'][coarse_full_mask].cpu().numpy()
        if 'scale0' in data:
            coarse_full_match_points0=coarse_full_match_points0 / data['scale0'][bs].cpu().numpy()[[1, 0]]
    else:
        coarse_full_match_points0=None
    
    if "detector_kpts0" in data:
        mask_keypoints= data['detector_b_ids'] == bs
        detector_keypoints0=data['detector_kpts0'][mask_keypoints].cpu().numpy()
    else:
        detector_keypoints0=None

    if coarse_full_match_points0 is None or detector_keypoints0 is None:
        figure = make_matching_plot(image0, image1, kpts0, kpts1, kpts0, kpts1, color, text)
    else:
        text.append(f'coarse_full_matches:{len(coarse_full_match_points0)}(blue)')
        text.append(f'detector_keypoints:{len(detector_keypoints0)}(yellow)')
        figure = make_matching_plot_show_more(image0,image1,kpts0,kpts1,kpts0,kpts1,color,text,keypoints_more_0=coarse_full_match_points0,keypoints_more_1=detector_keypoints0,show_keypoints=True,draw_matches=False)
    return figure


def draw_all_figures(data, config):
    """
    Args:
        data (dict)
        config (dict)
    Returns:
        figures (dict{plt.figure})
    # TODO: confidence mode plotting
    """
    figures = {'evaluation': []}
    # TODO: Parallel plotting @ang
    for bs in range(data['image0'].size(0)):
        figures['evaluation'].append(draw_sinlge_evaluation(data, bs))

    return figures
