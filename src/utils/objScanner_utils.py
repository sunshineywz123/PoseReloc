
import cv2
import numpy as np
import cv2
from pathlib import Path

def parse_3d_box(box_path):
    """ Read 3d box corners """
    with open(box_path, 'r') as f:
        lines = f.readlines()
    data = [float(e) for e in lines[1].strip().split(' ')]
    
    ex, ey, ez = data[3: 6]
    corner_3d = np.array([
        [ ex,  ey,  ez],
        [ ex, -ey,  ez],
        [ ex,  ey, -ez],
        [ ex, -ey, -ez],
        [-ex,  ey,  ez],
        [-ex, -ey,  ez],
        [-ex,  ey, -ez],
        [-ex, -ey, -ez]
    ]) * 0.5
    
    corner_3d_homo = np.hstack((corner_3d, np.ones((corner_3d.shape[0], 1))))
    
    return corner_3d, corner_3d_homo # [n, 3], [n, 4]

def read_box(box_file):
    assert Path(box_file).exists()

    with open(box_file, 'r') as f:
        lines = f.readlines()
    data = [float(e) for e in lines[1].strip().split(',')]
    position = data[:3]
    extent = data[3:6]
    quaternion = data[6:]

    return position, extent, quaternion

def read_transformation(trans_file):
    f = open(trans_file, 'r')
    line = f.readlines()[1]

    data = [float(e) for e in line.split(' ')]
    scale = np.array(data[0])
    rot_vec = np.array(data[1:4])
    trans_vec = np.array(data[4:])

    f.close()

    return scale, rot_vec, trans_vec

def get_refine_box(orig_box_file, trans_box_file):
    position, extent, quaternions = read_box(orig_box_file)

    corners_homo = np.array([
        [-extent[0], -extent[1], -extent[2], 1],
        [ extent[0], -extent[1], -extent[2], 1],
        [ extent[0], -extent[1],  extent[2], 1],
        [-extent[0], -extent[1],  extent[2], 1],
        [-extent[0],  extent[1], -extent[2], 1],
        [ extent[0],  extent[1], -extent[2], 1],
        [ extent[0],  extent[1],  extent[2], 1],
        [-extent[0],  extent[1],  extent[2], 1],
        [0, 0, 0, 1]
    ]).T #[4, 9]

    transformation = np.eye(4)
    scale, rot_vec, trans_vec = read_transformation(trans_box_file)
    rotation = cv2.Rodrigues(rot_vec)[0]
    transformation[:3, :3] = rotation
    transformation[:3, 3:] = trans_vec.reshape(3, 1)

    corners_homo[:3, :] *= 0.5
    refine_corners = corners_homo.copy()
    refine_corners[:3, :] *= scale
    trans_corners = transformation @ refine_corners

    return refine_corners[:3]


def parse_K(intrin_path):
    """ Read intrinsics"""
    with open(intrin_path, 'r') as f:
        lines = [line.rstrip('\n').split(' ')[1] for line in f.readlines()]
    
    fx, fy, cx, cy = list(map(float, lines))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return K, K_homo # [3, 3], [3, 4]
