import sys
sys.path.insert(0, '/home/wangzihao/code/sort_PoseReloc')

import os
import cv2
import json
import tqdm
import numpy as np
import os.path as osp

from src.utils.vis_utils import draw_3d_box
from src.utils import data_utils
from pathlib import Path
from transforms3d import affines, quaternions


def get_arkit_default_path(data_dir):
    video_file = osp.join(data_dir, 'Frames.m4v')
    
    color_dir = osp.join(data_dir, 'color')
    Path(color_dir).mkdir(parents=True, exist_ok=True)

    box_file = osp.join(data_dir, 'RefinedBox.txt')
    # box_file = osp.join(data_dir, 'Box.txt')
    assert Path(box_file).exists()

    out_pose_dir = osp.join(data_dir, 'poses')
    Path(out_pose_dir).mkdir(parents=True, exist_ok=True)
    pose_file = osp.join(data_dir, 'ARposes.txt')
    assert Path(pose_file).exists()
    
    intrin_file = osp.join(data_dir, 'Frames.txt')
    assert Path(intrin_file).exists()
    
    out_box_dir = osp.join(data_dir, 'bbox')
    Path(out_box_dir).mkdir(parents=True, exist_ok=True)

    orig_intrin_file = osp.join(data_dir, 'Frames.txt')
    final_intrin_file = osp.join(data_dir, 'intrinsics.txt')

    intrin_dir = osp.join(data_dir, 'intrin')
    Path(intrin_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        'video_file': video_file,
        'color_dir': color_dir,
        'box_path': box_file,
        'pose_file': pose_file,
        'out_box_dir': out_box_dir,
        'out_pose_dir': out_pose_dir,
        'orig_intrin_file': orig_intrin_file,
        'final_intrin_file': final_intrin_file,
        'intrin_dir': intrin_dir
    }
    
    return paths 
    

def get_bbox3d(box_path):
    assert Path(box_path).exists()
    with open(box_path, 'r') as f:
        lines = f.readlines()
    box_data = [float(e) for e in lines[1].strip().split(',')]
    ex, ey, ez = box_data[3: 6]
    bbox_3d = np.array([
        [ex,   ey,  ez],
        [ex,  -ey,  ez],
        [ex,   ey, -ez],
        [ex,  -ey, -ez],
        [-ex,  ey,  ez],
        [-ex, -ey,  ez],
        [-ex,  ey, -ez],
        [-ex, -ey, -ez]
    ]) * 0.5
    bbox_3d_homo = np.concatenate([bbox_3d, np.ones((8, 1))], axis=1)
    return bbox_3d, bbox_3d_homo


def get_K(intrin_file):
    assert Path(intrin_file).exists()
    with open(intrin_file, 'r') as f:
        lines = f.readlines()
    intrin_data = [line.rstrip('\n').split(':')[1] for line in lines]
    fx, fy, cx, cy = list(map(float, intrin_data))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return K, K_homo


def parse_box(box_path):
    with open(box_path, 'r') as f:
        lines = f.readlines()
    data = [float(e) for e in lines[1].strip().split(',')]
    position = data[:3]
    quaternion = data[6:]
    rot_mat = quaternions.quat2mat(quaternion)
    T_ow = affines.compose(position, rot_mat, np.ones(3))
    return T_ow


def reproj(K_homo, pose, points3d_homo):
    assert K_homo.shape == (3, 4)
    assert pose.shape == (4, 4)
    assert points3d_homo.shape[0] == 4 # [4 ,n]

    reproj_points = K_homo @ pose @ points3d_homo
    reproj_points = reproj_points[:]  / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points # [n, 2]


def parse_video(paths, downsample_rate=5, bbox_3d_homo=None):
    orig_intrin_file = paths['final_intrin_file']
    K, K_homo = get_K(orig_intrin_file)
    
    intrin_dir = paths['intrin_dir']
    cap = cv2.VideoCapture(paths['video_file'])
    index = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if index != 0 and index % downsample_rate == 0:
            img_name = osp.join(paths['color_dir'], '{}.png'.format(index))
            save_intrin_path = osp.join(intrin_dir, '{}.txt'.format(index))

            x0, y0, x1, y1 = np.loadtxt(osp.join(paths['out_box_dir'], '{}.txt'.format(index))).astype(int)
            box = np.array([x0, y0, x1, y1])
            resize_shape = np.array([y1 - y0, x1 - x0])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box, K, resize_shape)
            image_crop = data_utils.get_image_crop_resize(image, box, resize_shape)

            box_new = np.array([0, 0, x1-x0, y1-y0])  
            resize_shape = np.array([512, 512])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box_new, K_crop, resize_shape)
            image_crop = data_utils.get_image_crop_resize(image_crop, box_new, resize_shape)

            # cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 3)
            pose = np.loadtxt(osp.join(paths['out_pose_dir'], '{}.txt'.format(index)))
            reproj_crop = reproj(K_crop_homo, pose, bbox_3d_homo.T)
            x0_new, y0_new = reproj_crop.min(0)
            x1_new, y1_new = reproj_crop.max(0)
            box_new = np.array([x0_new, y0_new, x1_new, y1_new])
            
            np.savetxt(osp.join(paths['out_box_dir'], '{}.txt'.format(index)), box_new)
            cv2.imwrite(img_name, image_crop)
            full_img_dir = paths['color_dir'] + '_full'
            Path(full_img_dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(osp.join(full_img_dir, '{}.png'.format(index)), image)
            np.savetxt(save_intrin_path, K_crop)
            
        index += 1
    cap.release()


def data_process(data_dir, downsample_rate=5):
    paths = get_arkit_default_path(data_dir)
    with open(paths['orig_intrin_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 0 and l[0] != '#']
    eles = [[float(e) for e in l.split(',')] for l in lines]
    data = np.array(eles)
    fx, fy, cx, cy = np.average(data, axis=0)[2:]
    with open(paths['final_intrin_file'], 'w') as f:
        f.write('fx: {0}\nfy: {1}\ncx: {2}\ncy: {3}'.format(fx, fy, cx, cy))
    
    bbox_3d, bbox_3d_homo = get_bbox3d(paths['box_path'])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])

    with open(paths['pose_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        index = 0
        for line in tqdm.tqdm(lines):
            if len(line) == 0 or line[0] == '#':
                continue

            if index % downsample_rate == 0:
                eles = line.split(',') 
                data = [float(e) for e in eles]

                position = data[1:4]
                quaternion = data[4:]
                rot_mat = quaternions.quat2mat(quaternion)
                rot_mat = rot_mat @ np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ])

                T_ow = parse_box(paths['box_path'])
                T_cw = affines.compose(position, rot_mat, np.ones(3))
                T_wc = np.linalg.inv(T_cw)
                T_oc = T_wc @ T_ow
                pose_save_path = osp.join(paths['out_pose_dir'], '{}.txt'.format(index))
                box_save_path = osp.join(paths['out_box_dir'], '{}.txt'.format(index))
                reproj_box3d = reproj(K_homo, T_oc, bbox_3d_homo.T)

                x0, y0 = reproj_box3d.min(0)
                x1, y1 = reproj_box3d.max(0)
                reproj_box2d = np.array([x0, y0, x1, y1])
                np.savetxt(pose_save_path, T_oc)
                np.savetxt(box_save_path, reproj_box2d)
            index += 1
    
    parse_video(paths, downsample_rate, bbox_3d_homo)


def ln_data():
    data_root = './data/scan_data'
    orig_data_root = '/data/IKEA_Obj'

    orig_datasets = os.listdir(orig_data_root)
    for orig_dataset in orig_datasets:
        obj_data_root = osp.join(data_root, orig_dataset)
        sub_dirs = os.listdir(osp.join(orig_data_root, orig_dataset))
        obj_name = orig_dataset.split('-')[1]
        if not osp.isdir(obj_data_root):
            os.mkdir(obj_data_root)

            for sub_dir in sub_dirs:
                if obj_name not in sub_dir:
                    continue
                orig_path = osp.join(orig_data_root, orig_dataset, sub_dir)
                sub_dir_path = osp.join(obj_data_root, sub_dir)
                os.mkdir(sub_dir_path)

                os.system(f'ln -s {orig_path}/* {sub_dir_path}')
                print(f'=> ln {sub_dir_path}')
                

if __name__ == "__main__":
    # ln_data()
    data_root = './data/scan_data'
    seq_dirs = [
        # '0410-huiyuan-box', '0411-doveshampoo-others', '0412-pikachubowl-others',
        # '0413-juliecookies-box', '0414-babydiapers-others', '0415-captaincup-others',
        # '0416-tongyinoodles-others', '0417-kangshifunoodles-others', '0418-cookies1-others',
        # '0419-cookies2-others', '0420-liuliumei-others', '0421-cannedfish-others',
        # '0422-qvduoduo-box', '0423-oreo-box', '0425-vitacare-others',
        # '0426-prunes-others', '0427-raisins-others', '0430-newtolmeat-others',
        # '0432-sesamesedge-others', '0434-plum-others', '0436-sedge-others',
        # '0438-royal-others', '0439-nongfuspring-others', '0440-goatmilk-others',
        # '0442-hersheys-others', '0443-wheatbear-others',
        # '0444-weizen-others', '0445-pistonhead-others'
    ]
    for seq_dir in seq_dirs:
        data_dir = osp.join(data_root, seq_dir)
        subdirs = os.listdir(data_dir)
        for subdir in subdirs:
            if not subdir.startswith('.'):
                print('=> processing: ', subdir)
                data_process(osp.join(data_dir, subdir))