import argparse
import os
import os.path as osp
from itertools import combinations
import numpy as np
import scipy.spatial.distance as distance


from src.NeuralSfM import neuralSfM


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument(
        "--n_images",
        type=int,
        default=None,
        help="Only process a small subset of all images for debug",
    )
    parser.add_argument("--enable_post_optimization", type=bool, default=True)
    parser.add_argument("--use_ray", action='store_true')

    # Colmap realted
    parser.add_argument("--min_model_size", type=int, default=10)
    parser.add_argument("--filter_max_reproj_error", type=int, default=4)

    args = parser.parse_args()
    return args

def get_pairswise_distances(pose_files):
    Rs = []
    ts = []

    seqs_ids = {}
    for i in range(len(pose_files)):
        pose_file = pose_files[i]
        seq_name = pose_file.split('/')[-3]
        if seq_name not in seqs_ids.keys():
            seqs_ids[seq_name] = [i]     
        else:
            seqs_ids[seq_name].append(i)
         
    for pose_file in pose_files:
        pose = np.loadtxt(pose_file)
        R = pose[:3, :3]
        t = pose[:3, 3:]
        Rs.append(R)
        ts.append(t)
    
    Rs = np.stack(Rs, axis=0)
    ts = np.stack(ts, axis=0)

    Rs = Rs.transpose(0, 2, 1) # [n, 3, 3]
    ts = -(Rs @ ts)[:, :, 0] # [n, 3, 3] @ [n, 3, 1]

    dist = distance.squareform(distance.pdist(ts))
    trace = np.einsum('nji,mji->mn', Rs, Rs, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))

    return dist, dR, seqs_ids

def covis_from_pose(img_lists, pose_list, num_matched, min_rotation=10):
    img_type = img_lists[0].split('/')[-2]
    pose_lists = [img_file.replace(f'/{img_type}/', '/poses/').replace('.png', '.txt') for img_file in img_lists]
    dist, dR, seqs_ids = get_pairswise_distances(pose_list)

    valid = dR > min_rotation
    np.fill_diagonal(valid, False)
    dist = np.where(valid, dist, np.inf)

    pairs = []
    num_matched_per_seq = num_matched // len(seqs_ids.keys())
    for i in range(len(img_lists)):
        dist_i = dist[i]
        for seq_id in seqs_ids:
            ids = np.array(seqs_ids[seq_id])
            idx = np.argpartition(dist_i[ids], num_matched_per_seq * 2)[: num_matched_per_seq:2] 
            idx = ids[idx]
            idx = idx[np.argsort(dist_i[idx])]
            idx = idx[valid[i][idx]]

            for j in idx:
                name0 = img_lists[i]
                name1 = img_lists[j]

                pairs.append((name0, name1))
    
    return pairs

def main():
    args = parse_args()
    work_dir = args.work_dir
    n_images = args.n_images

    # Prepare data structure
    image_pth = osp.join(work_dir, "images")
    assert osp.exists(image_pth), f"{image_pth} is not exist!"
    img_names = sorted(os.listdir(image_pth))
    img_list = [osp.join(image_pth, img_name) for img_name in img_names if '._' not in img_name][:n_images]

    # generate image pairs:
    # exhauctive matching pairs:
    # NOTE: you can add covisible information to generate your own pairs to reduce the matching complexity
    pair_ids = list(combinations(range(len(img_list)), 2))
    img_pairs = []
    for pair_id in pair_ids:
        img_pairs.append(" ".join([img_list[pair_id[0]], img_list[pair_id[1]]]))
    
    colmap_configs = {}
    colmap_configs['min_model_size'] = args.min_model_size
    colmap_configs['filter_max_reproj_error'] = args.filter_max_reproj_error

    neuralSfM(
        img_list,
        img_pairs,
        work_dir=work_dir,
        enable_post_optimization=args.enable_post_optimization,
        use_ray=args.use_ray,
        colmap_configs=colmap_configs
    )


if __name__ == "__main__":
    main()

