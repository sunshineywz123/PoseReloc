from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid

from .geometry import warp_kpts

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config, data_source):
    """
    Update:
        data (dict):{
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        NOTE: for scannet dataset, there're 3 kinds of resolution {i, c, f}
        NOTE: for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR_BACKBONE']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None, [1, 0]] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None, [1, 0]] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check)
    # (points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]
    # TODO: 4'. filter out matches not retrievable with the window-size used for better coarse-spvs?

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse matching found!! {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['source'])) == 1, "Do not support mixed datasets training!"
    data_source = data['source'][0]
    if data_source in ['scannet', 'megadepth']:
        spvs_coarse(data, config, data_source)
    else:
        raise NotImplementedError


##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def spvs_fine(data, config, data_source):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['LOFTR_BACKBONE']['RESOLUTION'][1]
    radius = config['LOFTR_FINE']['WINDOW_SIZE'] // 2
    
    # change scale & radius adaptively
    # min_layer_id = min(config['LOFTR_BACKBONE']['RESNETFPN']['OUTPUT_LAYERS'])
    # rel_scale = int(log(config['LOFTR_BACKBONE']['RESOLUTION'][1], 2)) - min_layer_id
    # scale = scale / 2**rel_scale
    # radius = radius * 2**rel_scale

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    scale = scale * data['scale1'][b_ids][:, [1, 0]] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})


@torch.no_grad()
def spvs_fine_spp(data, config, data_source):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. get prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
    scale = data['hw0_i'][0] / data['hw0_c'][0]  # consider coarse level
    scale0 = scale * data['scale0'][b_ids][:, [1, 0]] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][b_ids][:, [1, 0]] if 'scale1' in data else scale
    mkpts0_c = torch.stack([i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]], dim=1) * scale0
    mkpts1_c = torch.stack([j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]], dim=1) * scale1

    # since detector is applied on data['image0'], a factor from image to depth might exist
    local_scale0 = 1 * data['scale0'][data['b_ids']][:, [1, 0]] if 'scale0' in data else 1
    mkpts0_f = mkpts0_c + data['i_associated_kpts_local'] * local_scale0
    mkpts1_f = mkpts1_c

    # warp from image0 to image1
    w_pt0_i = []
    for bs in range(data['image0'].size(0)):
        mask = b_ids == bs
        _, w_pt0_i_ = warp_kpts(
            mkpts0_f[mask][None],
            data['depth0'][bs][None], data['depth1'][bs][None],
            data['T_0to1'][bs][None], data['K0'][bs][None], data['K1'][bs][None]
        )
        w_pt0_i.append(w_pt0_i_[0])
    w_pt0_i = torch.cat(w_pt0_i, dim=0)

    # 2. Compute Gt
    scale = data['hw0_i'][0] / data['hw0_f'][0]  # consider fine level
    scale1 = scale * data['scale1'][b_ids][:, [1, 0]] if 'scale0' in data else scale
    radius = data['W'] // 2
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1
    expec_f_gt = (w_pt0_i - mkpts1_f) / scale1 / radius  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine(data, config):
    data_source = data['source'][0]
    if data_source in ['scannet', 'megadepth']:
        if config['LOFTR_MATCH_FINE']['DETECTOR'] == 'OnGrid':
            spvs_fine(data, config, data_source)
        elif config['LOFTR_MATCH_FINE']['DETECTOR'] in ['SuperPoint', 'SuperPointEC', 'SIFT'] or 'and grid' in config['LOFTR_MATCH_FINE']['DETECTOR']:
            spvs_fine_spp(data, config, data_source)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
