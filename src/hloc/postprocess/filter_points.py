from genericpath import exists
import cv2
import torch
import numpy as np
import os
import os.path as osp
from src.utils.colmap import read_write_model 


def filter_by_track_length(points3D, track_length):
    """ 
        Filter 3d points by track length.
        Return new pcds and corresponding point ids in origin pcds.
    """
    idxs_3d = list(points3D.keys())
    idxs_3d.sort()
    xyzs = np.empty(shape=[0, 3])
    points_idxs = np.empty(shape=[0], dtype=int)
    for i in range(len(idxs_3d)):
        idx_3d = idxs_3d[i]
        if len(points3D[idx_3d].point2D_idxs) < track_length:
            continue
        xyz = points3D[idx_3d].xyz.reshape(1, -1)
        xyzs = np.append(xyzs, xyz, axis=0)
        points_idxs = np.append(points_idxs, idx_3d)
    
    return xyzs, points_idxs


def get_3d_box_pose(bbox_path):
    """ Read box rotation and translation from box file generated by ObjectScanner"""
    from scipy.spatial.transform import Rotation

    def parse_bbox(bbox_path):
        with open(bbox_path) as f:
            lines = f.readlines()
        data = [float(e) for e in lines[1].strip().split(',')]
        return data
    
    bbox_data = np.array(parse_bbox(bbox_path))
    tvec, qvec, scale = bbox_data[:3], bbox_data[-4:], bbox_data[3:6]
    
    tvec = tvec.reshape(3, 1)
    rvec = Rotation.from_quat(qvec).as_rotvec().reshape(3, 1)
    scale = scale.reshape(3, )
    return tvec, rvec, scale


def get_3d_box(bbox_path):
    """ Get 3d box corners in canonical coordinate """
    tvec, rvec, scale = get_3d_box_pose(bbox_path)
    rotation = cv2.Rodrigues(rvec)[0]

    T = np.hstack((rotation, tvec))
    T = np.vstack((T, np.array([0, 0, 0, 1])))

    corner_in_cano = np.array([
        [-scale[0], -scale[0], -scale[0], -scale[0],  scale[0],  scale[0],  scale[0],  scale[0]],
        [-scale[1], -scale[1],  scale[1],  scale[1], -scale[1], -scale[1],  scale[1],  scale[1]],
        [-scale[2],  scale[2],  scale[2], -scale[2], -scale[2],  scale[2],  scale[2], -scale[2]],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]).T
    corner_in_cano = corner_in_cano[:, :3] * 0.5
    return corner_in_cano, T


def get_trans_box(trans_box_path):
    with open(trans_box_path, 'r') as f:
        lines = f.readlines() 
    
    data = [float(e) for e in lines[1].split(' ')]
    scale = np.array(data[0])
    rot_vec = np.array(data[1:4])
    trans_vec = np.array(data[4:])

    return scale, rot_vec, trans_vec


def trans_corner(orig_corner, trans_box_path):
    """
    @orig_corner: [n, 3]
    @trans_box_path: scale, rvec, tvec 

    return: 
        trans_corner [n, 3]
    """
    scale, rot_vec, trans_vec = get_trans_box(trans_box_path)

    corner_in_cano_homo = np.concatenate([orig_corner, np.ones((orig_corner.shape[0], 1))], axis=-1)
    corner_in_cano_homo = corner_in_cano_homo.T
    transformation = np.eye(4)
    transformation[:3, :3] = cv2.Rodrigues(rot_vec)[0]
    transformation[:3, 3:] = trans_vec.reshape(3, 1)

    trans_corner_in_cano_homo = transformation @ corner_in_cano_homo
    trans_corner_in_cano_homo[:3] /= trans_corner_in_cano_homo[3:]

    return trans_corner_in_cano_homo[:3].T


def filter_by_3d_box(points, points_idxs, box_path, trans_box_path=None):
    """ Filter 3d points by 3d box."""
    corner_in_cano, _ = get_3d_box(box_path)
    if trans_box_path is not None:
        corner_in_cano = trans_corner(corner_in_cano, trans_box_path)

    assert points.shape[1] == 3, "Input pcds must have shape (n, 3)"
    if not isinstance(points, torch.Tensor):
        points = torch.as_tensor(points, dtype=torch.float32)
    if not isinstance(corner_in_cano, torch.Tensor):
        corner_in_cano = torch.as_tensor(corner_in_cano, dtype=torch.float32)
    
    def filter_(bbox_3d, points):
        """
        @param bbox_3d: corners (8, 3)
        @param points: (n, 3)
        """
        v45 = bbox_3d[5] - bbox_3d[4]
        v40 = bbox_3d[0] - bbox_3d[4]
        v47 = bbox_3d[7] - bbox_3d[4]
        
        points = points - bbox_3d[4]
        m0 = torch.matmul(points, v45)
        m1 = torch.matmul(points, v40)
        m2 = torch.matmul(points, v47)
        
        cs = []
        for m, v in zip([m0, m1, m2], [v45, v40, v47]):
            c0 = 0 < m
            c1 = m < torch.matmul(v, v)
            c = c0 & c1
            cs.append(c)
        cs = cs[0] & cs[1] & cs[2]
        passed_inds = torch.nonzero(cs).squeeze(1)
        num_passed = torch.sum(cs)
        return num_passed, passed_inds, cs
    
    num_passed, passed_inds, keeps = filter_(corner_in_cano, points)
    
    xyzs_filtered = np.empty(shape=(0, 3), dtype=np.float32)
    for i in range(int(num_passed)):
        ind = passed_inds[i]
        xyzs_filtered = np.append(xyzs_filtered, points[ind, None], axis=0)
    
    filtered_xyzs = points[passed_inds]
    passed_inds = points_idxs[passed_inds]
    return filtered_xyzs, passed_inds

def filter_bbox(model_path, model_updated_save_path, box_path, box_trans_path=None):
    """ Filter 3d points by bbox, and save as colmap format """
    from src.utils.colmap.read_write_model import read_model, write_model
    cameras, images, points3D = read_model(model_path, ext='.bin')

    # Get 3D bbox:
    corner_in_cano, _ = get_3d_box(box_path)
    if box_trans_path is not None:
        corner_in_cano = trans_corner(corner_in_cano, box_trans_path)
    
    # Get pointcloud:
    pointclouds = []
    ids = []
    for id, point3D in points3D.items():
        pointclouds.append(point3D.xyz)
        ids.append(id)
    pointclouds = np.stack(pointclouds) # N*3
    ids = np.array(ids) # N
    
    # Filter 3D points by bbox
    def filter_(bbox_3d, points):
        """
        @param bbox_3d: corners (8, 3)
        @param points: (n, 3)
        """
        v45 = bbox_3d[5] - bbox_3d[4]
        v40 = bbox_3d[0] - bbox_3d[4]
        v47 = bbox_3d[7] - bbox_3d[4]
        
        points = points - bbox_3d[4]
        m0 = np.matmul(points, v45)
        m1 = np.matmul(points, v40)
        m2 = np.matmul(points, v47)
        
        cs = []
        for m, v in zip([m0, m1, m2], [v45, v40, v47]):
            c0 = 0 < m
            c1 = m < np.matmul(v, v)
            c = c0 & c1
            cs.append(c)
        cs = cs[0] & cs[1] & cs[2]
        passed_inds = np.nonzero(cs)
        num_passed = np.sum(cs)
        return num_passed, passed_inds, cs
    num_passed, passed_inds, keeped_mask = filter_(corner_in_cano, pointclouds)

    passed_ids = ids[~keeped_mask].tolist()

    # Update colmap model
    points3D_keeped = {}
    for id, point3D in points3D.items():
        if id in passed_ids:
            # Update images state!
            for img_id, point2D_idx in zip(point3D.image_ids.tolist(), point3D.point2D_idxs.tolist()):
                images[img_id].point3D_ids[point2D_idx] = -1
        else:
            # Keep!
            points3D_keeped[id] = point3D
    
    # TODO: Save updated colmap model
    if not osp.exists(model_updated_save_path):
        os.makedirs(model_updated_save_path, exist_ok=True)
    write_model(cameras, images, points3D_keeped, model_updated_save_path, ext='.bin')

# NOTE: Function duplicate! Because of old repo
def filter_3d(model_path, track_length, box_path, box_trans_path=None):
    """ Filter 3d points by tracke length and 3d box """
    points_model_path = osp.join(model_path, 'points3D.bin')
    points3D = read_write_model.read_points3d_binary(points_model_path)
   
    xyzs, points_idxs = filter_by_track_length(points3D, track_length)
    xyzs, points_idxs = filter_by_3d_box(xyzs, points_idxs, box_path, box_trans_path)

    return xyzs, points_idxs

def filter_track_length_and_bbox(model_path, track_length, box_path, box_trans_path=None):
    """ Filter 3d points by tracke length and 3d box """
    points_model_path = osp.join(model_path, 'points3D.bin')
    points3D = read_write_model.read_points3d_binary(points_model_path)
   
    xyzs, points_idxs = filter_by_track_length(points3D, track_length)

    return xyzs, points_idxs

def filter_track_length(model_path, track_length):
    """ Filter 3d points by track length """
    points_model_path = osp.join(model_path, 'points3D.bin')
    points3D = read_write_model.read_points3d_binary(points_model_path)
   
    xyzs, points_idxs = filter_by_track_length(points3D, track_length)
    return xyzs, points_idxs

def merge(xyzs, points_idxs, dist_threshold=1e-3):
    """ 
    Merge points which are close to others. ({[x1, y1], [x2, y2], ...} => [mean(x_i), mean(y_i)])
    """
    from scipy.spatial.distance import pdist, squareform
    
    if not isinstance(xyzs, np.ndarray):
        xyzs = np.array(xyzs)

    dist = pdist(xyzs, 'euclidean')
    distance_matrix = squareform(dist)
    close_than_thresh = distance_matrix < dist_threshold

    ret_points_count = 0 # num of return points
    ret_points = np.empty(shape=[0, 3]) # pcds after merge
    ret_idxs = {} # {new_point_idx: points idxs in Points3D}

    points3D_idx_record = [] # points that have been merged
    for j in range(distance_matrix.shape[0]):
        idxs = close_than_thresh[j] 

        # TODO: check
        if np.isin(points_idxs[idxs], points3D_idx_record).any():
            continue

        points = np.mean(xyzs[idxs], axis=0) # new point
        ret_points = np.append(ret_points, points.reshape(1, 3), axis=0)
        ret_idxs[ret_points_count] = points_idxs[idxs]
        ret_points_count += 1
        
        points3D_idx_record = points3D_idx_record + points_idxs[idxs].tolist()
    
    return ret_points, ret_idxs


    