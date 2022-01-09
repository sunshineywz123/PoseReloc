import torch

@torch.no_grad()
def fine_supervision(data, config):
    """
    Update:
        data (dict): {
            "expec_f_gt": [M, 2]
        }
    """
    coarse_scale, fine_scale = list(config['loftr']['loftr_backbone']['resolution'])
    radius = config['loftr']['loftr_fine']['window_size'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    coarse_scale =coarse_scale * data['query_image_scale'][b_ids][:, [1, 0]] if 'query_image_scale' in data else fine_scale 
    fine_scale = fine_scale * data['query_image_scale'][b_ids][:, [1, 0]] if 'query_image_scale' in data else fine_scale

    mkpts_query = (
        torch.stack([j_ids % data["q_hw_c"][1], j_ids // data["q_hw_c"][1]], dim=1)
        * coarse_scale
    ) # [M, 2]

    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1
    expec_f_gt = (data['fine_location_matrix_gt'][b_ids, i_ids, j_ids] - mkpts_query) / fine_scale / radius  # [M, 2]
    
    #FIXME: check here!
    data.update({"expec_f_gt": expec_f_gt})