import argparse
import os
import os.path as osp
from shutil import copyfile, copytree, rmtree
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--scannet_database_dir", type=str, default="/data/scannet/output")
    # parser.add_argument("--retrival_results_file", type=str, required=True)
    parser.add_argument(
        "--aim_dir",
        type=str,
        default='/nas/users/hexingyi/scannet_onposeform',
    )

    args = parser.parse_args()
    return args

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

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

if __name__ == "__main__":
    args = parse_args()
    aim_dir = args.aim_dir
    scannet_database_dir = args.scannet_database_dir
    os.makedirs(aim_dir, exist_ok=True)

    scene_names_all = os.listdir(scannet_database_dir)
    scene_names_all = [scene_name for scene_name in scene_names_all if "scene" in scene_name]

    for global_id, scene_name in tqdm(enumerate(scene_names_all), total=len(scene_names_all)):
        assert len(str(global_id)) <= 4
        parsed_global_id = "0" * (4 - len(str(global_id))) + str(global_id) # 0023-query_name
        os.makedirs(osp.join(aim_dir, parsed_global_id + '-' + scene_name), exist_ok=True) # Obj path
        os.makedirs(osp.join(aim_dir, parsed_global_id + '-' + scene_name, scene_name+'-1'), exist_ok=True) # Seq path
        seq_name = osp.join(aim_dir, parsed_global_id + '-' + scene_name, scene_name+'-1')

        # Copy pose dir
        nan_list = []
        pose_dir = osp.join(seq_name, "poses_ba")
        if osp.exists(pose_dir):
            rmtree(pose_dir)
        os.makedirs(pose_dir, exist_ok=True)
        src_pose_dir = osp.join(scannet_database_dir, scene_name, 'pose')
        for pose_name in os.listdir(src_pose_dir):
            # NOTE: pose in scannet dataset is camera to world, need to inverse before colmap Mapping
            pose_cam2world = np.loadtxt(osp.join(src_pose_dir, pose_name))
            pose_world2cam = np.linalg.inv(pose_cam2world)
            # if np.nan in pose_cam2world:
            #     nan_list += [osp.splitext(pose_name)[0]]
            #     continue
            try:
                qvec = rotmat2qvec(pose_world2cam[:3,:3])
            except:
                nan_list += [osp.splitext(pose_name)[0]]
                continue
            np.savetxt(osp.join(pose_dir, pose_name), pose_world2cam)

        # Copy color image
        color_dir = osp.join(seq_name, "color")
        if osp.exists(color_dir):
            rmtree(color_dir)
        os.makedirs(color_dir, exist_ok=True)
        src_color_dir = osp.join(scannet_database_dir, scene_name, 'color')
        for color_name in os.listdir(src_color_dir):
            if osp.splitext(color_name)[0] in nan_list:
                continue
            copyfile(osp.join(src_color_dir, color_name), osp.join(color_dir, color_name))

        # Build intrinsic dir
        intrinisc_dir = osp.join(seq_name, 'intrin_ba')
        os.makedirs(intrinisc_dir, exist_ok=True)
        # extract intrinsics for all images in this scene
        K_homo = np.loadtxt(osp.join(scannet_database_dir, scene_name, 'intrinsic', 'intrinsic_color.txt'))
        K = K_homo[:3,:3] # 3*3
        for img_name in os.listdir(color_dir):
            intrinsic_name = img_name.replace('.jpg', '.txt')
            np.savetxt(osp.join(intrinisc_dir, intrinsic_name), K)
        
    # Make fake bbox?