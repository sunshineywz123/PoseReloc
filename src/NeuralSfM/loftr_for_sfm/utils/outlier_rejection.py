"""utils for outlier rejection"""


def filter_outlier(data, config):
    """
    Update:
        data (dict):{
            'mconf' (torch.Tensor): [M, 1]
            'm_bids' (torch.Tensor): [M]
            'mkpts0_f' (torch.Tensor): [M, 2]
            'mkpts1_f' (torch.Tensor): [M, 2]
            'outlier_mask' (torch.Tensor): [M]
            }
    """
    if config['LOFTR_FINE']['REJECTOR']['ENABLE']:
        assert 'mconf_f' in data, 'Fine-level rejection enabled but no `mconf_f` in `data`.'
    else:
        return

    _inlier = (data['mconf_f'] >= config['LOFTR_FINE']['REJECTOR']['THR'])
    data.update(
        {field: data[field][_inlier] for field in ['mconf', 'mkpts0_f', 'mkpts1_f', 'm_bids']}
    )
    data.update({'outlier_mask': ~_inlier})
