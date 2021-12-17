import numpy as np
import cv2
from loguru import logger
import pydegensac

def extract_geo_model_inliers(mkpts0, mkpts1, mconfs,
                              geo_model, ransac_method, pixel_thr, max_iters, conf_thr,
                              K0=None, K1=None):
    # TODO: early return if len(mkpts) < min_candidates
    
    if geo_model == 'E':
        f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
        pixel_thr = pixel_thr / f_mean

        mkpts0, mkpts1 = map(lambda x: normalize_ketpoints(*x), [(mkpts0, K0), (mkpts1, K1)])
    
    if ransac_method == 'RANSAC':
        if geo_model == 'E':
            E, mask = cv2.findEssentialMat(mkpts0, 
                                           mkpts1,
                                           np.eye(3),
                                           threshold=pixel_thr, 
                                           prob=conf_thr, 
                                           method=cv2.RANSAC)
        elif geo_model == 'F':
            F, mask = cv2.findFundamentalMat(mkpts0,
                                             mkpts1,
                                             method=cv2.FM_RANSAC,
                                             ransacReprojThreshold=pixel_thr,
                                             confidence=conf_thr,
                                             maxIters=max_iters)
    elif ransac_method == 'DEGENSAC':
        assert geo_model == 'F'
        F, mask = pydegensac.findFundamentalMatrix(mkpts0,
                                                   mkpts1,
                                                   px_th=pixel_thr,
                                                   conf=conf_thr,
                                                   max_iters=max_iters)
    elif ransac_method == 'MAGSAC':
        raise NotImplementedError()
    else:
        raise ValueError()
    
    if mask is not None:
        mask = mask.astype(bool).flatten()
    else:
        mask = np.full_like(mconfs, True, dtype=np.bool)
    return mask

def agg_groupby_2d(keys, vals, agg='avg'):
    """
    Args:
        keys: (N, 2) 2d keys
        vals: (N,) values to average over
        agg: aggregation method
    Returns:
        dict: {key: agg_val}
    """
    assert agg in ['avg', 'sum']
    unique_keys, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group_sums = np.bincount(group, weights=vals)
    values = group_sums if agg == 'sum' else group_sums / counts
    return dict(zip(map(tuple, unique_keys), values))


class Match2Kpts(object):
    """extract all possible keypoints for each image from all image-pair matches"""
    def __init__(self, matches, names, name_split='-', cov_threshold=0):
        self.names = names
        self.matches = matches
        self.cov_threshold = cov_threshold
        self.name2matches = {name: [] for name in names}
        for k in matches.keys():
            try:
                name0, name1 = k.split(name_split)
            except ValueError as _:
                name0, name1 = k.split('-')
            self.name2matches[name0].append((k, 0))
            self.name2matches[name1].append((k, 1))
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            name = self.names[idx]
            kpts = np.concatenate([self.matches[k][:, [2*id, 2*id+1, 4]]
                        for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold], 0)
            return name, kpts
        elif isinstance(idx, slice):
            names = self.names[idx]
            try:
                kpts = [np.concatenate([self.matches[k][:, [2*id, 2*id+1, 4]]
                            for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold], 0) for name in names]
            except:
                kpts = []
                for name in names:
                    kpt = [self.matches[k][:, [2*id, 2*id+1, 4]]
                            for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold]
                    if len(kpt) != 0:
                        kpts.append(np.concatenate(kpt,0))
                    else:
                        kpts.append(np.empty((0,3)))
                        logger.warning(f"no keypoints in image:{name}")
            return list(zip(names, kpts))
        else:
            raise TypeError(f'{type(self).__name__} indices must be integers')