import argparse
import os
import os.path as osp
from shutil import copyfile, rmtree
from tqdm import tqdm
import cv2
import numpy as np
import torch
from scipy.io import loadmat

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--inloc_database_dir", type=str, default="/nas/users/hexingyi/InLoc/InLoc")
    parser.add_argument("--retrival_results_file", type=str, required=True)
    parser.add_argument(
        "--aim_dir",
        type=str,
        default='/nas/users/hexingyi/InLoc_onposeform',
    )

    args = parser.parse_args()
    return args

def parse_retrieval(path):
    retrival_dict = {} # query_img_name: [db_images]
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r = p.split()
            if q in retrival_dict:
                retrival_dict[q].append(r)
            else:
                retrival_dict[q] = [r]
    return retrival_dict

def get_scan_pose(dataset_dir, rpath):
    split_image_rpath = rpath.split('/')
    floor_name = split_image_rpath[-3]
    scan_id = split_image_rpath[-2]
    image_name = split_image_rpath[-1]
    building_name = image_name[:3]

    path = osp.join(
        dataset_dir, 'database/alignments', floor_name,
        f'transformations/{building_name}_trans_{scan_id}.txt')
    with open(path) as f:
        raw_lines = f.readlines()

    P_after_GICP = np.array([
        np.fromstring(raw_lines[7], sep=' '),
        np.fromstring(raw_lines[8], sep=' '),
        np.fromstring(raw_lines[9], sep=' '),
        np.fromstring(raw_lines[10], sep=' ')
    ])

    return P_after_GICP

def interpolate_scan(scan, kp):
    h, w, c = scan.shape
    kp = kp / np.array([[w-1, h-1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(
        scan, kp, align_corners=True, mode='bilinear')[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode='nearest')[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    kp3d = interp.T.numpy()
    valid = valid.numpy()
    return kp3d, valid

def ransac_PnP(K, pts_2d, pts_3d, scale=1, pnp_reprojection_error=5):
    """ solve pnp """
    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
    K = K.astype(np.float64)

    pts_3d *= scale
    state = None
    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            K,
            dist_coeffs,
            reprojectionError=pnp_reprojection_error,
            iterationsCount=10000,
            flags=cv2.SOLVEPNP_EPNP,
        )
        # _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, dist_coeffs)

        rotation = cv2.Rodrigues(rvec)[0]

        tvec /= scale
        pose = np.concatenate([rotation, tvec], axis=-1)
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

        if inliers is None:
            inliers = np.array([]).astype(np.bool)
        state = True

        return pose, pose_homo, inliers, state
    except cv2.error:
        print("CV ERROR")
        state = False
        return np.eye(4)[:3], np.eye(4), np.array([]).astype(np.bool), state

def convert_pose2T(pose):
    # pose: [R: 3*3, t: 3]
    R, t = pose
    return np.concatenate(
        [np.concatenate([R, t[:, None]], axis=1), [[0, 0, 0, 1]]], axis=0
    )  # 4*4

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

if __name__ == "__main__":
    args = parse_args()
    aim_dir = args.aim_dir
    inloc_database_dir = args.inloc_database_dir
    os.makedirs(aim_dir, exist_ok=True)

    # Parse retrival results:
    retrival_results_file_path = args.retrival_results_file 
    assert osp.exists(retrival_results_file_path)
    retrival_dict = parse_retrieval(retrival_results_file_path) # query image name(bag name): [db_image_names]

    for global_id, (q_name, db_imgs_path) in tqdm(enumerate(retrival_dict.items()), total=len(retrival_dict)):
        image_name = osp.basename(q_name)
        image_basename = osp.splitext(image_name)[0]
        assert len(str(global_id)) <= 4
        parsed_global_id = "0" * (4 - len(str(global_id))) + str(global_id) # 0023-query_name
        os.makedirs(osp.join(aim_dir, parsed_global_id + '-' + image_basename), exist_ok=True) # Obj path
        os.makedirs(osp.join(aim_dir, parsed_global_id + '-' + image_basename, image_basename+'-1'), exist_ok=True) # Seq path
        seq_name = osp.join(aim_dir, parsed_global_id + '-' + image_basename, image_basename+'-1')

        color_dir = osp.join(seq_name, "color")
        if osp.exists(color_dir):
            rmtree(color_dir)
        os.makedirs(color_dir, exist_ok=True)
        intrinisc_dir = osp.join(seq_name, 'intrin_ba')
        os.makedirs(intrinisc_dir, exist_ok=True)
        pose_dir = osp.join(seq_name, "poses_ba")
        os.makedirs(pose_dir, exist_ok=True)
        for i, db_img_path in enumerate(db_imgs_path):
            # Move image to Onepose friendly format
            src_ = osp.join(inloc_database_dir, db_img_path)
            ext = osp.splitext(src_)[1]
            dst_ = osp.join(color_dir, f"{i}{ext}")
            copyfile(src_, dst_)

            # Parse intrinsics:
            height, width = cv2.imread(src_).shape[:2]
            cx = 0.5 * width
            cy = 0.5 * height
            focal_length = 4032. * 28. / 36 # Fixed intrinsics
            K = np.array([[focal_length, 0, cx],
                          [0, focal_length, cy],
                          [0, 0,            1]])
            np.savetxt(osp.join(intrinisc_dir, f"{i}.txt"), K)

            # Parse extrinsics:
            pose = get_scan_pose(inloc_database_dir, db_img_path) # world_temp to real_world

            # Solve PnP to get world temp to image
            scan = loadmat(osp.join(inloc_database_dir, db_img_path + '.mat'))["XYZcut"]
            x = np.linspace(1, width-2, num=width-2, dtype=np.int)
            y = np.linspace(1, height-2, num=height-2, dtype=np.int)
            xv, yv = np.meshgrid(x,y)
            coord2D = np.stack([xv,yv], axis=-1) # H*W*2
            coord2D = coord2D.reshape(-1, 2) # (H*W) * 2
            n_sample = 6000
            sample_index = np.random.randint(coord2D.reshape(-1,2).shape[0], size=n_sample)
            coord2D_sampled = coord2D[sample_index]
            coord3D, valid = interpolate_scan(scan, coord2D_sampled)
            coord3D_valid = coord3D[valid]
            coord2D_valid = coord2D_sampled[valid]
            try:
                import pycolmap
                cfg = {
                    'model': 'SIMPLE_PINHOLE',
                    'width': width,
                    'height': height,
                    'params': [focal_length, cx, cy]
                }
                ret = pycolmap.absolute_pose_estimation(coord2D_valid, coord3D_valid, cfg, 15.00)
                qvec = ret['qvec']
                tvec = ret['tvec']
                pose_worldtemp2camera = convert_pose2T([qvec2rotmat(qvec), tvec])

            except:
                _, pose_worldtemp2camera, inliers, state = ransac_PnP(K, coord2D_valid, coord3D_valid, scale=1000, pnp_reprojection_error=15)
                import ipdb; ipdb.set_trace()

            pose = pose_worldtemp2camera @ np.linalg.inv(pose)

            np.savetxt(osp.join(pose_dir, f'{i}.txt'), pose)
    # Make fake bbox?