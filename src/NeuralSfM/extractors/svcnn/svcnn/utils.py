import torch
from pathlib import Path
from torch import nn
import numpy as np
import cv2

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5  # calc down-sampled keypoints positions
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):  # keep the `scores` shape unchanged
        """ Suppress points whose score isn't the maximum within the local patch.
        """
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)  # max: 1, non-max: 0
    for _ in range(2):  # ???
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]

def top_k_keypoints(keypoints, scores, k: int, img_h: int, img_w: int, mode: str):
    """
    Args:
        keypoints (torch.Tensor): (n_kpts, 2)
        scores (torch.Tensor): (n_kpts, )
    """
    if k >= len(keypoints):
        # Randomly pad keypoints to k with score = 0
        if mode == 'train':
            padded_kpts, padded_scores = pad_keypoints_random_v2(keypoints, scores, img_h, img_w, k)
            return padded_kpts, padded_scores
        else:
            return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores

def top_k_keypoints_with_descriptor(keypoints, scores, descriptors, k: int, img_h: int, img_w: int, mode: str):
    """
    Args:
        keypoints (torch.Tensor): (n_kpts, 2)
        scores (torch.Tensor): (n_kpts, )
        descriptors (torch.Tensor): (dim, n_kpts)
    """
    if k > len(keypoints):
        # Randomly pad keypoints to k with score = 0
        if mode == 'train':
            padded_kpts, padded_scores = pad_keypoints_random_v2(keypoints, scores, img_h, img_w, k)
            padded_desc = torch.cat([descriptors, torch.zeros([len(descriptors), len(padded_kpts)-len(keypoints)],
                                                              dtype=descriptors.dtype, device=descriptors.device)], 1)
            return padded_kpts, padded_scores, padded_desc
        else:
            return keypoints, scores, descriptors
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores, descriptors[:, indices]

def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2*radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
            scores[:, None], width, 1, radius, divisor_override=1)
    ar = torch.arange(-radius, radius+1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(
            scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
            scores[:, None], kernel_x.transpose(2, 3), padding=radius)
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints

def refine_with_harris(keypoints, image, harris_radius):
    """
    keypoints: (1, (kps_num, 2)), [y,x]
    """
    image_numpy = (image.cpu().numpy()*255)[0, 0]
    corners = np.flip(keypoints[0].cpu().numpy(), axis=1)
    blockSize = 2
    apertureSize = 3
    k = 0.04
    dst = cv2.cornerHarris(image_numpy, blockSize, apertureSize, k)

    h, w = dst.shape
    for i in range(len(corners)):
        x, y = corners[i]
        left = max(0, x-harris_radius)
        right = min(w, x+harris_radius+1)
        top = max(0, y-harris_radius)
        down = min(h, y+harris_radius+1)
        new_corner = np.argmax(dst[top:down, left:right])
        new_x, new_y = left+new_corner % (right-left), top+new_corner//(right-left)
        if dst[new_y, new_x] != dst[y, x]:
            corners[i][0] = new_x
            corners[i][1] = new_y

    # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners = cv2.cornerSubPix(image_numpy, np.float32(corners), (harris_radius, harris_radius), (-1, -1), criteria)
    corners = np.flip(corners, axis=1).copy()
    return [torch.from_numpy(corners).to(keypoints[0].device)]

def pad_keypoints_random_v2(keypoints, scores, img_h: int, img_w: int, n_target_kpts: int):
    """ Pad the given keypoints to the target #kpts. The padded kpts shouldn't overlap with
    existing kpts.
    Args:
        keypoints (torch.Tensor): sorted keypoints with shape (n_kpts, 2). 
            (sorted is not required)
    Returns:
        padded_kpts (torch.Tensor): (n_target_kpts, 2).
        padded_scores (torch.Tensor): (n_target_kpts,)
    """
    device = keypoints.device
    dtype = keypoints.dtype
    n_pad = n_target_kpts - keypoints.shape[0]
    # TODO: Optimization
    while n_pad > 0:
        # TODO: add torch.Generator
        rand_kpts_x = torch.randint(0, img_w, (n_pad, ), dtype=dtype, device=device)
        rand_kpts_y = torch.randint(0, img_h, (n_pad, ), dtype=dtype, device=device)
        rand_kpts = torch.stack([rand_kpts_y, rand_kpts_x], 1)

        exist = (rand_kpts[:, None, :] == keypoints[None, :, :]).all(-1).any(1)  # (n_pad, )
        kept_kpts = rand_kpts[~exist]  # (n_kept, 2)
        n_pad -=len(kept_kpts)
        
        if len(kept_kpts) > 0:
            keypoints = torch.cat([keypoints, kept_kpts], 0)
            scores = torch.cat([scores, torch.zeros(len(kept_kpts), dtype=scores.dtype, device=device)], 0)
    return keypoints, scores

def quadratic_refinement(keypoints, scores):
    di_filter = torch.tensor(
        [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]).view(1, 1, 3, 3)
    dj_filter = torch.tensor(
        [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]).view(1, 1, 3, 3)

    dii_filter = torch.tensor(
        [[0, 1., 0], [0, -2., 0], [0, 1., 0]]).view(1, 1, 3, 3)
    dij_filter = 0.25 * torch.tensor(
        [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]).view(1, 1, 3, 3)
    djj_filter = torch.tensor(
        [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]).view(1, 1, 3, 3)

    scores = scores[:, None]  # B x 1 x H x W
    dii = torch.nn.functional.conv2d(scores, dii_filter.to(scores), padding=1)
    dij = torch.nn.functional.conv2d(scores, dij_filter.to(scores), padding=1)
    djj = torch.nn.functional.conv2d(scores, djj_filter.to(scores), padding=1)
    det = dii * djj - dij * dij

    inv_hess_00 = djj / det
    inv_hess_01 = -dij / det
    inv_hess_11 = dii / det

    di = torch.nn.functional.conv2d(scores, di_filter.to(scores), padding=1)
    dj = torch.nn.functional.conv2d(scores, dj_filter.to(scores), padding=1)

    delta_i = -(inv_hess_00 * di + inv_hess_01 * dj)
    delta_j = -(inv_hess_01 * di + inv_hess_11 * dj)
    delta = torch.stack([delta_i, delta_j], -1)[:, 0]  # B x H x W x 2
    valid = torch.all(delta.abs() < 0.5, -1)
    delta = torch.where(
        valid[..., None].expand_as(delta), delta, delta.new_zeros(1))

    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = delta[i][tuple(kpts.t())]
        # print(torch.all(delta == 0., -1).float().mean().item())
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints