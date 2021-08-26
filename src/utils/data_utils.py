import cv2
import torch
import numpy as np


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


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
        

def get_image_crop_resize(image, box, resize_shape):
    """Crop image according to the box, and resize the cropped image to resize_shape
    @param image: the image waiting to be cropped
    @param box: [x0, y0, x1, y1]
    @param resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    scale = np.array([box[2] - box[0], box[3] - box[1]])
    
    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    image_crop = cv2.warpAffine(image, trans_crop, (resize_w, resize_h), flags=cv2.INTER_LINEAR)
    return image_crop


def get_K_crop_resize(box, K_orig, resize_shape):
    """Update K (crop an image according to the box, and resize the cropped image to resize_shape) 
    @param box: [x0, y0, x1, y1]
    @param K_orig: [3, 3] or [3, 4]
    @resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    scale = np.array([box[2] - box[0], box[3] - box[1]]) # w, h
    
    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)

    if K_orig.shape == (3, 3):
        K_orig_homo = np.concatenate([K_orig, np.zeros((3, 1))], axis=-1)
    else:
        K_orig_homo = K_orig.copy()
    assert K_orig_homo.shape == (3, 4)

    K_crop_homo = trans_crop_homo @ K_orig_homo # [3, 4]
    K_crop = K_crop_homo[:3, :3]
    
    return K_crop, K_crop_homo