import numpy as np


def project(xyz, K, RT, need_depth=False):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T)
    xyz += RT[:, 3:].T
    depth = xyz[:, 2:].flatten()
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    if need_depth:
        return xy, depth
    else:
        return xy


class Visualizer(object):
    def __init__(self, vis_path):
        self.vis_path = vis_path

    def set_new_seq(self, seq_name):
        from vis3d.vis3d import Vis3D
        self.vis3d = Vis3D(self.vis_path, seq_name)

    def set_scene_id(self, scene_id):
        self.vis3d.set_scene_id(scene_id)

    def compute_epipolar_error(self, kpts0, kpts1, T_0to1, K0, K1):
        def to_homogeneous(points):
            return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        kpts0 = to_homogeneous(kpts0)
        kpts1 = to_homogeneous(kpts1)

        t0, t1, t2 = T_0to1[:3, 3]
        t_skew = np.array([
            [0, -t2, t1],
            [t2, 0, -t0],
            [-t1, t0, 0]
        ])
        E = t_skew @ T_0to1[:3, :3]

        Ep0 = kpts0 @ E.T  # N x 3
        p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
        Etp1 = kpts1 @ E  # N x 3
        d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
                          + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
        return d

    def add_kpt_2d3d_simple(self, fid, query_img, mkps2d, mkps3d, pose_gt, K, meta=None, kps2d=None, kps3d=None,
                            metrics=None):
        from arscan.utils import project

        if len(mkps3d) == 0:
            return
        mkps3d_proj = project(mkps3d, K, pose_gt)
        if kps3d is not None:
            kps3d_full = project(kps3d, K, pose_gt)
        else:
            kps3d_full = mkps3d_proj

        if kps2d is None:
            kps2d = mkps2d

        rep_error = np.linalg.norm(mkps2d - mkps3d_proj, axis=1)
        if metrics is not None:
            metrics.update({'1-rep_err': rep_error})
        else:
            metrics = {'1-rep_err': rep_error}

        self.vis3d.set_scene_id(fid)
        self.vis3d.add_keypoint_correspondences(query_img,
                                                query_img,
                                                mkps2d, mkps3d_proj,
                                                unmatched_kpts0=kps2d,
                                                unmatched_kpts1=kps3d_full,
                                                metrics=metrics,
                                                meta=meta)

    def add_kpt_2d3d(self, fid, query_img, kps2d, match01, kps3d, q2d_to_db3d, pose_gt, K, kps3d_arry=None):
        from arscan.utils import project
        valid = np.where(match01 != -1)
        mkps2d = kps2d[valid[0]]
        kpts_3d = []
        for pts_pair in q2d_to_db3d:
            kpts_3d.append(kps3d[pts_pair[1]][0])
        kpts_3d = np.array(kpts_3d)
        mkps3d_proj = project(kpts_3d, K, pose_gt)
        if kps3d_arry is not None:
            kps3d_full = project(kps3d_arry, K, pose_gt)
        else:
            kps3d_full = mkps3d_proj
        # rep_error = np.abs(mkps2d - mkps3d_proj)
        rep_error = np.linalg.norm(mkps2d - mkps3d_proj, axis=1)
        metrics = {'rep_err': rep_error}
        self.vis3d.set_scene_id(fid)
        self.vis3d.add_keypoint_correspondences(query_img,
                                                query_img,
                                                mkps2d, mkps3d_proj,
                                                unmatched_kpts0=kps2d,
                                                unmatched_kpts1=kps3d_full,
                                                metrics=metrics,
                                                meta=None)

    def add_kpt_corr(self, fid, img0, img1, kps0, kps1, match01=None, T_0to1=None, K=None, K2=None,
                     kpt2d_proj=None, metrics=None):
        if match01 is None:
            mkpts0 = kps0
            mkpts1 = kps1
        else:
            valid = np.where(match01 != -1)
            match01 = match01[valid[0]]
            mkpts0 = kps0[valid[0]]
            mkpts1 = kps1[match01]

        if T_0to1 is not None:
            if K2 is None:
                K2 = K
            epipolar_error = self.compute_epipolar_error(mkpts0, mkpts1, T_0to1, K, K2)
        else:
            epipolar_error = None

        if kpt2d_proj is not None:
            rep_error = np.abs(kpt2d_proj - mkpts0)
        else:
            rep_error = None

        if metrics is None:
            metrics = {'epp_err': epipolar_error,
                       'rep_err': rep_error}
            if epipolar_error is None and rep_error is None:
                metrics = None

        self.vis3d.set_scene_id(fid)

        self.vis3d.add_keypoint_correspondences(img0,
                                                img1,
                                                mkpts0, mkpts1,
                                                unmatched_kpts0=kps0,
                                                unmatched_kpts1=kps1,
                                                metrics=metrics,
                                                meta=None)


def put_text(img, inform_text, color=None):
    import cv2
    fontScale = 1
    if color is None:
        color = (255, 0, 0)
    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    img = cv2.putText(img, inform_text, org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
    return img
