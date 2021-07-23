
import numpy as np

def parse_3d_box(box_path):
    """ Read 3d box corners """
    with open(box_path, 'r') as f:
        lines = f.readlines()
    data = [float(e) for e in lines[1].strip().split(',')]
    
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


def parse_K(intrin_path):
    """ Read intrinsics"""
    with open(intrin_path, 'r') as f:
        lines = [line.rstrip('\n').split(':')[1] for line in f.readlines()]
    
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
