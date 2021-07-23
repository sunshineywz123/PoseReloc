import cv2
import torch
import numpy as np


def pad_keypoints2d_random(keypoints, features, scores, img_h, img_w, n_target_kpts):
    dtype = keypoints.dtype
    
    n_pad = n_target_kpts - keypoints.shape[0]
    if n_pad < 0:
        keypoints = keypoints[:n_target_kpts] # [n_target_kpts, 2]
        features = features[:, :n_target_kpts] # [dim, n_target_kpts]
        scores = scores[:n_target_kpts] # [n_target_kpts, 1]
    else:
        while n_pad > 0:
            random_kpts_x = torch.randint(0, img_w, (n_pad, ), dtype=dtype)
            random_kpts_y = torch.randint(0, img_h, (n_pad, ), dtype=dtype)
            rand_kpts = torch.stack([random_kpts_y, random_kpts_x], dim=1)
            
            exist = (rand_kpts[:, None, :] == keypoints[None, :, :]).all(-1).any(1) # (n_pad, )
            kept_kpts = rand_kpts[~exist] # (n_kept, 2)
            n_pad -= len(kept_kpts)
            if len(kept_kpts) > 0:
                keypoints = torch.cat([keypoints, kept_kpts], 0)
                scores = torch.cat([scores, torch.zeros(len(kept_kpts), 1, dtype=scores.dtype)], dim=0)
                features = torch.cat([features, torch.ones(features.shape[0], len(kept_kpts))], dim=1)
    
    return keypoints, features, scores


def pad_keypoints3d_random(keypoints, features, scores, n_target_kpts):
    n_pad = n_target_kpts - keypoints.shape[0]
    
    if n_pad < 0:
        keypoints = keypoints[:n_target_kpts] # [n_target_kpts, 3]
        features = features[:, :n_target_kpts] # [dim, n_target_kpts]
        scores = scores[:n_target_kpts] # [n_target_kpts, 1] 
    else:
        while n_pad > 0:
            rand_kpts_x = torch.rand(n_pad, 1) - 0.5 # zero_mean
            rand_ktps_y = torch.rand(n_pad, 1) - 0.5 # zero_mean
            rand_kpts_z = torch.rand(n_pad, 1) - 0.5 # zero_mean
            rand_kpts = torch.cat([rand_kpts_x, rand_ktps_y, rand_kpts_z], dim=1)
            
            exist = (rand_kpts[:, None, :] == keypoints[None, :, :]).all(-1).any(1)
            kept_kpts = rand_kpts[~exist] # [n_kept, 3]
            n_pad -= len(kept_kpts)
            
            if len(kept_kpts) > 0:
                keypoints = torch.cat([keypoints, kept_kpts], dim=0)
                features = torch.cat([features, torch.ones((features.shape[0], kept_kpts.shape[0]))], dim=-1)
                scores = torch.cat([scores, torch.zeros((len(kept_kpts), 1), dtype=scores.dtype)], dim=0)
    
    return keypoints, features, scores


def reshape_assign_matrix(assign_matrix, orig_shape2d, orig_shape3d, shape2d, shape3d, pad=True):
    """ Reshape assign matrix (from 2xk to nxm)"""
    assign_matrix = assign_matrix.long()
    
    if pad:
        conf_matrix = torch.zeros(shape2d, shape3d, dtype=torch.int16)
        
        valid = (assign_matrix[0] < shape2d) & (assign_matrix[1] < shape3d)
        assign_matrix = assign_matrix[:, valid]

        conf_matrix[assign_matrix[0], assign_matrix[1]] = 1
        conf_matrix[orig_shape2d:] = -1
        conf_matrix[:, orig_shape3d:] = -1
    else:
        conf_matrix = torch.zeros(orig_shape2d, orig_shape3d, dtype=torch.int16)
        
        valid = (assign_matrix[0] < shape2d) & (assign_matrix[1] < shape3d)
        conf_matrix = conf_matrix[:, valid]
        
        conf_matrix[assign_matrix[0], assign_matrix[1]] = 1
    
    return conf_matrix
        