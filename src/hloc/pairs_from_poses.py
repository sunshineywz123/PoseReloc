import numpy as np
import scipy.spatial.distance as distance


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


def covis_from_pose(img_lists, covis_pairs_out, num_matched, max_rotation, do_ba=False):
    img_type = img_lists[0].split('/')[-2]
    if not do_ba:
        pose_lists = [img_file.replace(f'/{img_type}/', '/poses_ba/').replace('.png', '.txt') for img_file in img_lists]
    else:
        pose_lists = [img_file.replace(f'/{img_type}/', '/poses/').replace('.png', '.txt') for img_file in img_lists]
    dist, dR, seqs_ids = get_pairswise_distances(pose_lists)

    min_rotation = 10
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

    with open(covis_pairs_out, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
