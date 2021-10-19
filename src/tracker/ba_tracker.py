import numpy as np
import cv2
from .vis_utils import Visualizer, project
from src.utils.vis_utils import reproj, ransac_PnP
from src.models.matchers.nn.nearest_neighbour import NearestNeighbour
device = 'cuda'
from matplotlib import pyplot as plt


class BATracker:
    def __init__(self, cfg):
        self.kf_frames = dict()
        self.query_frames = dict()
        self.id = 0
        self.last_kf_id = -1
        self.vis = Visualizer('./vis')
        # self.vis.set_new_seq('tracking_test')
        self.extractor = self.load_extractor_model(cfg, cfg.model.extractor_model_path)
        self.matcher = NearestNeighbour()
        self.pose_list = []

        self.kpt2ds = [] # coordinate for kpt
        self.kpt2d_available_list = []
        self.kpt2d_descs = [] # may be change to descriptor list
        self.kpt2d_fids = [] # fid for kpt
        self.cams = [] # list of cam params
        self.kf_kpt_index_dict = dict() # kf_id -> [2d_id_start, 2d_id_end]
        # self.db_3d_dict = dict() # db_3d_id -> 3d_id
        self.db_3d_list = np.array([])

        self.kpt3d_list = []
        self.kpt2d3d_ids = [] # 3D ids of each 2D keypoint

    def load_extractor_model(self, cfg, model_path):
        """ Load extractor model(SuperGlue) """
        from src.models.extractors.SuperPoint.superpoint_v1 import SuperPoint
        from src.hloc.extract_features import confs
        from src.utils.model_io import load_network

        extractor_model = SuperPoint(confs[cfg.network.detection]['conf'])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path)

        return extractor_model

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def cm_degree_5_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        return translation_distance, angular_distance

    def kpt_flow_track(self, im_kf, im_query, kpt2d_last):
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # kpt_last = np.array(, dtype=np.float32)
        kpt_last = np.expand_dims(kpt2d_last, axis=1)
        kpt_new, status, err = cv2.calcOpticalFlowPyrLK(im_kf, im_query, kpt_last, None, **lk_params)
        valid_id = np.where(status.flatten() == 1)
        kpt_new = np.squeeze(kpt_new, axis=1)
        return kpt_new, valid_id

    def _update_3d_ids(self, kf_db_ids):
        self.db_3d_list = np.concatenate([self.db_3d_list, kf_db_ids])

    def add_kf(self, kf_info_dict):
        self.kf_frames[self.id] = kf_info_dict
        self.pose_list.append(kf_info_dict['pose_pred'])

        if len(self.kpt2ds) == 0:
            # update camera params
            self.cams = np.array([self.get_cam_param(kf_info_dict['K'], kf_info_dict['pose_pred'])])

            # initialize 2D keypoints
            kpt_pred = kf_info_dict['kpt_pred']

            n_kpt = kpt_pred['keypoints'].shape[0]
            self.kpt2ds = kpt_pred['keypoints'] # [n_2d, 2]
            self.kpt2d_match = np.zeros([n_kpt], dtype=int) # [n_2d, ]
            self.kpt2d_descs = kpt_pred['descriptors'].transpose() # [n_2d, n_dim]
            self.kpt2d_fids = np.ones([n_kpt], dtype=int) * self.id

            # initialize camera_list
            self.kf_kpt_index_dict[self.id] = (0, n_kpt-1)

            # init 3D points & 2D-3D relationship
            self.kpt3d_list = np.array(kf_info_dict['mkpts3d'])
            self.kpt2d3d_ids = np.ones([n_kpt], dtype=int) * -1

            kf_3d_ids = np.arange(0, len(kf_info_dict['mkpts3d']))
            self.kpt2d3d_ids[kf_info_dict['valid_query_id']] = kf_3d_ids

            kf_db_ids = kf_info_dict['kpt3d_ids']
            # create mapping from DB id to kpt3d id
            self._update_3d_ids(kf_db_ids)
        else:
            # update camera params
            kf_cam = np.array([self.get_cam_param(kf_info_dict['K'], kf_info_dict['pose_pred'])])
            self.cams = np.concatenate([self.cams, kf_cam], axis=0)

            # update 2D keypoints
            kpt_pred = kf_info_dict['kpt_pred']
            n_kpt = kpt_pred['keypoints'].shape[0]
            self.kpt2ds = np.concatenate([self.kpt2ds, kpt_pred['keypoints']], axis=0) # [n_2d, 2]
            self.kpt2d_match = np.concatenate([self.kpt2d_match, np.zeros([n_kpt])], axis=0) # [n_2d, ]
            self.kpt2d_descs = np.concatenate([self.kpt2d_descs, kpt_pred['descriptors'].transpose()], axis=0) # [n_2d, ]
            self.kpt2d_fids = np.concatenate([self.kpt2d_fids, np.ones([n_kpt]) * self.id])

            # initialize camera_list
            start_id = self.kf_kpt_index_dict[self.last_kf_id][-1] + 1
            self.kf_kpt_index_dict[self.id] = (start_id, start_id + n_kpt - 1)

            # Find non-duplicate 3d ids in kf 3d points and in database 3d points
            kpt3d_db_ids = self.db_3d_list
            kf_db_ids = kf_info_dict['kpt3d_ids']
            intersect_kpts = np.intersect1d(kpt3d_db_ids, kf_db_ids)
            mask_kf_3d_exist = np.in1d(kf_db_ids, intersect_kpts) # [bool, ]
            mask_kpt3d_found = np.in1d(kpt3d_db_ids, intersect_kpts) # [bool, ]

            kf_3d_ids_ndup = np.where(mask_kf_3d_exist == False)[0] # non-duplicate kf 3d keypoint ids
            kf_kpt3ds_new = kf_info_dict['mkpts3d'][kf_3d_ids_ndup] # non-duplicate kf 3d keypoint

            # Update 2D-3D relationship
            kf_kpt2d3d_id = np.ones([n_kpt]) * -1
            valid_qid = kf_info_dict['valid_query_id']

            # For duplicate parts, 3D ids copy from existing ids
            kpt_3d_ids_dup = np.where(mask_kpt3d_found == True)[0]
            kf_3d_ids_dup = np.where(mask_kf_3d_exist == True)[0]
            valid_id_dup = valid_qid[kf_3d_ids_dup]
            kf_kpt2d3d_id[valid_id_dup] = np.arange(0, len(self.kpt3d_list))[kpt_3d_ids_dup]

            # For non-duplicate parts, 3D ids are created
            valid_id_ndup = valid_qid[kf_3d_ids_ndup]
            kpt3d_start_id = len(self.kpt3d_list)
            kf_kpt2d3d_id[valid_id_ndup] = np.arange(kpt3d_start_id, kpt3d_start_id + len(kf_kpt3ds_new))
            kf_kpt2d3d_id = np.asarray(kf_kpt2d3d_id, dtype=int)
            self.kpt2d3d_ids = np.concatenate([self.kpt2d3d_ids, kf_kpt2d3d_id], axis=0)

            # Update 3D keypoints
            self.kpt3d_list = np.concatenate([self.kpt3d_list, kf_kpt3ds_new], axis=0)

            # update mapping from DB id to kpt3d id
            kf_db_ids_new = kf_db_ids[kf_3d_ids_ndup] # non-duplicate kf 3d keypoint db id
            self._update_3d_ids(kf_db_ids_new)

        self.last_kf_id = self.id
        self.id += 1

    def frame_visualization(self, kpt2ds, kpt2d3d_ids, kpt3d_list, cams_f):
        for kf_id, kf_info in self.kf_frames.items():
            kf_cam_parm = cams_f[kf_id]
            K, kf_pose_pred = self.get_cam_params_back(kf_cam_parm)

            # Get 2D points
            kpt_idx_start, kpt_idx_end = self.kf_kpt_index_dict[kf_id]
            kpt_idx = np.arange(kpt_idx_start, kpt_idx_end + 1)
            kf_kps2d = kpt2ds[kpt_idx]

            # Get 3D points
            kf_2d3d_ids = kpt2d3d_ids[kpt_idx]
            kf_kpts3d = kpt3d_list[kf_2d3d_ids[np.where(kf_2d3d_ids != -1)]]
            kf_kpts_proj = project(kf_kpts3d, K, kf_pose_pred[:3])

            # Load image and visualization
            kf_im_path = kf_info['im_path']
            kf_im = cv2.imread(kf_im_path)

            from matplotlib import pyplot as plt
            plt.close()
            plt.imshow(kf_im)
            plt.plot(kf_kps2d[:, 0], kf_kps2d[:, 1], 'r+')
            plt.plot(kf_kpts_proj[:, 0], kf_kpts_proj[:, 1], 'b+')
            plt.show()

    def cuda2cpu(self, pred_detection_cuda):
        return {k: v[0].cpu().numpy() for k, v in pred_detection_cuda.items()}

    def apply_match(self, kpt_pred0, kpt_pred1):
        import torch
        data = {}
        for k in kpt_pred0.keys():
            data[k + '0'] = kpt_pred0[k]
        for k in kpt_pred1.keys():
            data[k + '1'] = kpt_pred1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device) for k, v in data.items()}
        matching_result = self.matcher(data)
        return matching_result

    def _triangulatePy(self, P1, P2, kpt2d_1, kpt2d_2):
        import scipy.linalg
        point3d = []
        for p1, p2 in zip(kpt2d_1, kpt2d_2):
            A = np.zeros([4, 4])
            A[0, :] = p1[0] * P1[2, :] - P1[0, :]
            A[1, :] = p1[1] * P1[2, :] - P1[1, :]
            A[2, :] = p2[0] * P2[2, :] - P2[0, :]
            A[3, :] = p2[1] * P2[2, :] - P2[1, :]

            U, a, Vh = scipy.linalg.svd(np.dot(A.T, A))
            X = -1 * U[:, 3]
            point3d.append(X[:3] / X[3])

        return np.array(point3d)

    def apply_triangulation(self, K, Tcw1, Tcw2, kpt2d_1, kpt2d_2):
        proj_mat1 = np.dot(K, np.linalg.inv(Tcw1)[:3, :])
        proj_mat2 = np.dot(K, np.linalg.inv(Tcw2)[:3, :])
        point_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, kpt2d_1.transpose(), kpt2d_2.transpose()).T
        point_3d_w = point_4d[:, :3] / np.repeat(point_4d[:, 3], 3).reshape(-1, 3)
        # point_3d_w2 = triangulatePy(proj_mat1, proj_mat2, kpt2d_1, kpt2d_2)
        return point_3d_w

    def motion_prediction(self):
        from transforms3d.euler import mat2euler, euler2mat
        pose0 = self.pose_list[-2]
        pose1 = self.pose_list[-1]

        trans0 = pose0[:3, 3]
        trans1 = pose1[:3, 3]

        rot0 = np.array(mat2euler(pose0[:3, :3]))
        rot1 = np.array(mat2euler(pose1[:3, :3]))
        trans_t = 2 * trans0 - trans1
        rot_t = 2 * rot0 - rot1
        pose_new = np.eye(4)

        pose_new[:3, :3] = euler2mat(rot_t[0], rot_t[1], rot_t[2])
        pose_new[:, 3] = trans_t
        return pose_new

    def flow_track(self, frame_info_dict):
        if self.id == self.last_kf_id:
            return
        # Load image
        kf_frame_info = self.kf_frames[self.last_kf_id]
        im_kf = cv2.imread(kf_frame_info['im_path'], cv2.IMREAD_GRAYSCALE)

        # Get initial pose with 2D-2D match from optical flow
        im_query = cv2.imread(frame_info_dict['im_path'], cv2.IMREAD_GRAYSCALE)
        mkpts2d_query, valid_ids = self.kpt_flow_track(im_kf, im_query, kf_frame_info['mkpts2d'])
        kpt3ds_kf = kf_frame_info['mkpts3d'][valid_ids]

        # Solve PnP to find initial pose
        pose_init, pose_init_homo, inliers = ransac_PnP(frame_info_dict['K'], mkpts2d_query, kpt3ds_kf)
        trans_dist, rot_dist = self.cm_degree_5_metric(pose_init_homo, frame_info_dict['pose_gt'])
        print(f"\nFlow pose error:{trans_dist} - {rot_dist}")
        # # Visualize correspondence
        # kpt2d_rep_q = project(kpt3ds_kf, frame_info_dict['K'],  frame_info_dict['pose_gt'][:3])
        # T_0to1 = np.dot(kf_frame_info['pose_gt'], np.linalg.inv(frame_info_dict['pose_gt']))
        # mkpt2ds_kf = kf_frame_info['mkpts2d'][valid_ids]
        # # kpt2d_rep_kf = project(kpt3ds, kf_frame_info['K'], kf_frame_info['pose_gt'][:3])
        # self.vis.add_kpt_corr(self.id, im_query, im_kf, mkpts2d_query, mkpt2ds_kf, kpt2d_proj=kpt2d_rep_q,
        #                       T_0to1=T_0to1, K=frame_info_dict['K'])
        return pose_init_homo

    def test_ba(self):
        import argparse
        import os
        import sys
        sys.path.append('/home/zhangsiyu/repos/PoseReloc/DeepLM')
        import torch
        import BACore
        import numpy as np

        from BAProblem.rotation import AngleAxisRotatePoint
        from BAProblem.loss import SnavelyReprojectionError
        from BAProblem.io import LoadBALFromFile
        from TorchLM.solver import Solve

        from time import time

        # parser = argparse.ArgumentParser(description='Bundle adjuster')
        # parser.add_argument('--balFile', default='data/problem-1723-156502-pre.txt')
        # parser.add_argument('--device', default='cuda')  # cpu/cuda
        # args = parser.parse_args()
        device = 'cuda'
        filename = '/home/zhangsiyu/repos/PoseReloc/DeepLM/data/problem-49-7776-pre.txt'

        # Load BA data
        points, cameras, features, ptIdx, camIdx = LoadBALFromFile(filename)

        # Optionally use CUDA
        points, cameras, features, ptIdx, camIdx = points.to(device), \
                                                   cameras.to(device), features.to(device), ptIdx.to(device), camIdx.to(
            device)

        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time()
        # optimize
        Solve(variables=[points, cameras],
              constants=[features],
              indices=[ptIdx, camIdx],
              fn=SnavelyReprojectionError,
              numIterations=15,
              numSuccessIterations=15)
        t2 = time()

        print("Time used %f secs." % (t2 - t1))

    def apply_ba(self, kpt2ds, kpt2d3d_ids, kpt2d_fids, kpt3d_list, cams):
        from DeepLM.BAProblem.loss import SnavelyReprojectionError
        from DeepLM.TorchLM.solver import Solve
        import torch
        device = 'cuda'
        points = torch.tensor(kpt3d_list, device=device, dtype=torch.float64, requires_grad=False)
        cameras = torch.tensor(cams, device=device, dtype=torch.float64, requires_grad=False)
        valid2d_idx = np.where(kpt2d3d_ids != -1)[0]
        features = torch.tensor(kpt2ds[valid2d_idx], device=device, dtype=torch.float64, requires_grad=False)
        ptIdx = torch.tensor(kpt2d3d_ids[valid2d_idx], device=device, dtype=torch.int64, requires_grad=False)
        camIdx = torch.tensor(kpt2d_fids[valid2d_idx], device=device, dtype=torch.int64, requires_grad=False)

        ################ DISPLAY AND VALIDATE INPUTS #########################
        features_ = features.cpu().numpy()
        ptIdx_ = ptIdx.cpu().numpy()
        camIdx_ = camIdx.cpu().numpy()
        cameras_ = cameras.cpu().numpy()
        points_ = points.cpu().numpy()
        rep_error = []
        for i in range(len(features)):
            pts2d = features_[i]
            pts3d = points_[ptIdx_[i]]
            cam = cameras_[camIdx_[i]]
            K, pose = self.get_cam_params_back(cam)
            rep_2d = project([pts3d], K, pose[:3])
            rep_error.append(np.linalg.norm(pts2d - rep_2d))

        print(f'Input stat:\n'
              f'- min:{np.min(rep_error)}\n'
              f'- max:{np.max(rep_error)}\n'
              f'- med:{np.median(rep_error)}')
        ################ DISPLAY AND VALIDATE INPUTS #########################

        # Display Initial Reprojection Error by frame
        kpt2ds_np = kpt2ds[valid2d_idx]
        kpt3d_idx = kpt2d3d_ids[valid2d_idx]
        camera_idx = np.asarray(kpt2d_fids[valid2d_idx], dtype=int)
        kpt3ds = kpt3d_list
        cams_np = cams[camera_idx]
        for frame_idx in np.unique(camera_idx):
            kpt_idx = np.where(camera_idx == frame_idx)[0]
            # kpt_idx = kpt_idx[np.where(kpt3d_idx[kpt_idx] > len(self.kpt3d_list))]
            kps2d = kpt2ds_np[kpt_idx]
            kps3d = kpt3ds[kpt3d_idx[kpt_idx]]
            kps_cam = cams_np[kpt_idx]
            K, pose_mat = self.get_cam_params_back(kps_cam[0])
            kps_rep = project(kps3d, K, pose_mat[:3])
            kps_rep_error = np.linalg.norm(kps2d - kps_rep, axis=1)
            print(f'Frame:{frame_idx}\n'
                  f'- min:{np.min(kps_rep_error)}\n'
                  f'- max:{np.max(kps_rep_error)}\n'
                  f'- med:{np.median(kps_rep_error)}')

        points, cameras, features, ptIdx, camIdx = points.to(device), \
                                                   cameras.to(device), features.to(device), ptIdx.to(device), camIdx.to(
            device)

        if device == 'cuda':
            torch.cuda.synchronize()

        # optimize
        Solve(variables=[points, cameras],
              constants=[features],
              indices=[ptIdx, camIdx],
              fn=SnavelyReprojectionError,
              numIterations=15,
              numSuccessIterations=15)

        points_opt_np = points.cpu().detach().numpy()
        cam_opt_np = cameras.cpu().detach().numpy()

        # Display Optimized Reprojection Error by frame
        kpt2ds_np = kpt2ds[valid2d_idx]
        kpt3d_idx = kpt2d3d_ids[valid2d_idx]
        camera_idx = np.asarray(kpt2d_fids[valid2d_idx], dtype=int)
        kpt3ds = points_opt_np
        cams_np = cam_opt_np[camera_idx]
        for frame_idx in np.unique(camera_idx):
            kpt_idx = np.where(camera_idx == frame_idx)[0]
            # kpt_idx = kpt_idx[np.where(kpt3d_idx[kpt_idx] > len(self.kpt3d_list))]
            kps2d = kpt2ds_np[kpt_idx]
            kps3d = kpt3ds[kpt3d_idx[kpt_idx]]
            kps_cam = cams_np[kpt_idx]
            K, pose_mat = self.get_cam_params_back(kps_cam[0])
            kps_rep = project(kps3d, K, pose_mat[:3])
            kps_rep_error = np.linalg.norm(kps2d - kps_rep, axis=1)
            print(f'Frame:{frame_idx}\n'
                  f'- min:{np.min(kps_rep_error)}\n'
                  f'- max:{np.max(kps_rep_error)}\n'
                  f'- med:{np.median(kps_rep_error)}')

        return points_opt_np, cam_opt_np

    def get_cam_params_back(self, cam_params):
        """ Convert BAL format to frame parameter to matrix form"""
        r_vec = cam_params[:3]
        t = cam_params[3:6]
        f = cam_params[6]
        k1 = cam_params[7]
        k2 = cam_params[8]
        K = np.array(
            [[f, 0, k1],
             [0, f, k2],
             [0, 0, 1]])
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = cv2.Rodrigues(r_vec)[0]
        pose_mat[:3, 3] = t
        return K, pose_mat

    def get_cam_param(self, K, pose):
        """ Convert frame parameter to BAL format"""
        f = K[0, 0]
        k1 = K[0, 2]
        k2 = K[1, 2]
        t = pose[:3, 3]
        R = cv2.Rodrigues(pose[:3, :3])[0]
        return np.concatenate([R.flatten(), t, [f, k1, k2]])

    def track(self, frame_info_dict):
        if len(self.pose_list) >= 2:
            pose_init = self.motion_prediction()
        else:
            pose_init = self.flow_track(frame_info_dict)

        # Load image
        kf_frame_info = self.kf_frames[self.last_kf_id]
        im_kf = cv2.imread(kf_frame_info['im_path'], cv2.IMREAD_GRAYSCALE)

        # self.frame_visualization()

        # Extract and match 2D keypoints
        # inp = frame_info_dict['data']['image'].cuda()
        # frame_info_dict.pop('data')
        # kpt2ds_pred_query = self.cuda2cpu(self.extractor(inp))
        kpt2ds_pred_query = frame_info_dict['kpt_pred']
        kpt2ds_pred_query.pop('scores')

        # Get KF 2D keypoints from data
        kpt_idx_start, kpt_idx_end = self.kf_kpt_index_dict[self.last_kf_id]
        kpt_idx = np.arange(kpt_idx_start, kpt_idx_end+1)
        kpt2ds_pred_kf = \
            { 'keypoints': self.kpt2ds[kpt_idx],
              'descriptors': self.kpt2d_descs[kpt_idx].transpose()}
        # kpt2ds_pred_kf = kf_frame_info['kpt_pred']

        # Apply match
        match_results = self.apply_match(kpt2ds_pred_kf, kpt2ds_pred_query)
        match_kq = match_results['matches0'][0].cpu().numpy()
        valid = np.where(match_kq != -1)
        mkpts2d_kf = kpt2ds_pred_kf['keypoints'][valid]
        mkpts2d_query = kpt2ds_pred_query['keypoints'][match_kq[valid]]
        kpt_idx_valid = kpt_idx[valid]

        # Update
        kpt2ds_match_f = np.copy(self.kpt2d_match)
        kpt2d3d_ids_f = np.copy(self.kpt2d3d_ids)

        # Update 2D inform
        n_kpt_q = len(mkpts2d_query)
        kpt2ds_f = np.concatenate([self.kpt2ds, mkpts2d_query]) # update 2D keypoints
        kpt2ds_match_f[valid] += 1 # update 2D match
        kpt2ds_match_f = np.concatenate([kpt2ds_match_f, np.ones([n_kpt_q])])

        # Check 2D-3D correspondence
        kf_2d_3d_ids = self.kpt2d3d_ids[kpt_idx_valid]
        kpt_idx_wo3d = np.where(kf_2d_3d_ids == -1)[0] # local index of point without 3D index
        mkpts2d_kf_triang = mkpts2d_kf[kpt_idx_wo3d]
        mkpts2d_query_triang = mkpts2d_query[kpt_idx_wo3d]

        # Triangulation
        Tco_kf = np.linalg.inv(kf_frame_info['pose_pred'])
        Tco_query = np.linalg.inv(pose_init)
        kpt3ds_triang = self.apply_triangulation(frame_info_dict['K'],
                                                 Tco_kf, Tco_query,
                                                 mkpts2d_kf_triang, mkpts2d_query_triang)

        # Remove triangulation points with extremly large error
        kpt2d_rep_kf = project(kpt3ds_triang, kf_frame_info['K'],  kf_frame_info['pose_gt'][:3])
        kpt2d_rep_query = project(kpt3ds_triang, frame_info_dict['K'],  frame_info_dict['pose_gt'][:3])

        rep_diff_q = np.linalg.norm(kpt2d_rep_query - mkpts2d_query_triang, axis=1)
        rep_diff_kf = np.linalg.norm(kpt2d_rep_kf - mkpts2d_kf_triang, axis=1)
        triang_rm_idx_q = np.where(rep_diff_q > 20)[0]
        triang_rm_idx_kf = np.where(rep_diff_kf > 20)[0]
        triang_rm_idx = np.unique(np.concatenate([triang_rm_idx_q, triang_rm_idx_kf]))
        triang_keep_idx = np.array([i for i in range(len(kpt2d_rep_query))
                                    if i not in triang_rm_idx]) # index over mkpts2d_q

        mkpts2d_kf_triang = mkpts2d_kf_triang[triang_keep_idx]
        mkpts2d_query_triang = mkpts2d_query_triang[triang_keep_idx]
        kpt2d_rep_kf = kpt2d_rep_kf[triang_keep_idx]
        kpt2d_rep_query = kpt2d_rep_query[triang_keep_idx]

        ########### Visualize 2D-2D match and initial 3D points ##########
        # T_0to1 = np.dot(kf_frame_info['pose_gt'], np.linalg.inv(frame_info_dict['pose_gt']))
        # im_query = cv2.imread(frame_info_dict['im_path'], cv2.IMREAD_GRAYSCALE)
        # self.vis.set_new_seq('match2d_qk')
        # self.vis.add_kpt_corr(self.id, im_query, im_kf, mkpts2d_query_triang, mkpts2d_kf_triang,
        #                       kpt2d_proj=kpt2d_rep_query,
        #                       T_0to1=T_0to1, K=frame_info_dict['K'])
        #
        # self.vis.set_new_seq('match_repo_q')
        # self.vis.add_kpt_corr(self.id, im_query, im_query, mkpts2d_query_triang, kpt2d_rep_query)
        # self.vis.set_new_seq('match_repo_kf')
        # self.vis.add_kpt_corr(self.id, im_kf, im_kf, mkpts2d_kf_triang, kpt2d_rep_query)
        #
        # from matplotlib import pyplot as plt
        # plt.close()
        # plt.imshow(im_query)
        # plt.plot(kpt2d_rep_query[:, 0], kpt2d_rep_query[:, 1], 'bo')
        # plt.plot(mkpts2d_query_triang[:, 0], mkpts2d_query_triang[:, 1], 'r+')
        # plt.show()
        ########### Visualize 2D-2D match and initial 3D points ##########

        # Update 2D-3D correspondence
        query_2d3d_ids = np.ones(n_kpt_q) * -1
        kpt_idx_w3d = np.where(kf_2d_3d_ids != -1)[0]
        query_2d3d_ids[kpt_idx_w3d] = kf_2d_3d_ids[kpt_idx_w3d]
        mkpts2d_query_exist = mkpts2d_query[kpt_idx_w3d]
        mkpts2d_kf_exist = mkpts2d_kf[kpt_idx_w3d]

        ########## Visualize 2D-2D match and existing 3D points ##############
        # kpt3d_exist = self.kpt3d_list[ kf_2d_3d_ids[kpt_idx_w3d]]
        # kpt2d_rep_exist = project(kpt3d_exist, frame_info_dict['K'],  frame_info_dict['pose_gt'][:3])
        #
        # self.vis.set_new_seq('repo_match_pts')
        # self.vis.add_kpt_corr(self.id, im_query, im_kf, mkpts2d_query_exist, mkpts2d_kf_exist,
        #                       kpt2d_proj=kpt2d_rep_exist,
        #                       T_0to1=T_0to1, K=frame_info_dict['K'])
        #
        # plt.close()
        # plt.imshow(im_query)
        # plt.plot(kpt2d_rep_exist[:, 0], kpt2d_rep_exist[:, 1], 'bo')
        # plt.plot(mkpts2d_query_exist[:, 0], mkpts2d_query_exist[:, 1], 'r+')
        # plt.show()
        ########## Visualize 2D-2D match and existing 3D points ##############

        # Update correspondence for newly triangulated points
        kpt3d_start_id = len(self.kpt3d_list)
        query_2d3d_ids[kpt_idx_wo3d[triang_keep_idx]] = np.arange(kpt3d_start_id, kpt3d_start_id + len(triang_keep_idx))
        query_2d3d_ids = np.asarray(query_2d3d_ids, dtype=int)
        kpt2d3d_ids_f = np.concatenate([self.kpt2d3d_ids, query_2d3d_ids])

        # Add 3D points
        kpt3d_list_f = np.concatenate([self.kpt3d_list, kpt3ds_triang[triang_keep_idx]])
        cams_f = np.concatenate([self.cams,  [self.get_cam_param(frame_info_dict['K'], pose_init)]])
        kpt2d_fids_f = np.concatenate([self.kpt2d_fids, np.ones([n_kpt_q]) * self.id])

        # ###################  Calculate Reprojection Error and visualization  ######################################
        # kpt_idxs = np.where(kpt2d_fids_f == 1)[0]
        # start_idx = np.min(kpt_idxs)
        # kpt_idxs = kpt_idxs[np.where(kpt2d3d_ids_f[kpt_idxs] != -1)[0]]
        #
        # kpt3d_full = kpt3d_list_f[kpt2d3d_ids_f[kpt_idxs]]
        # kpt2d_full = kpt2ds_f[kpt_idxs]
        # # rep3d_full = project(kpt3d_full, frame_info_dict['K'], frame_info_dict['pose_gt'][:3])
        # rep3d_full = project(kpt3d_full, frame_info_dict['K'], pose_init[:3])
        # kps_error_full = np.linalg.norm(kpt2d_full- rep3d_full, axis=1)
        # print(f'Full points:'
        #       f'- min:{np.min(kps_error_full)}\n'
        #       f'- max:{np.max(kps_error_full)}\n'
        #       f'- med:{np.median(kps_error_full)}')
        #
        # kps2d_triang_ids = np.where(kpt2d3d_ids_f[start_idx:] > len(self.kpt3d_list))[0] + start_idx
        # kpt3d_triang_ids = kpt2d3d_ids_f[kps2d_triang_ids]
        # rep3d = project(kpt3d_list_f[kpt3d_triang_ids], frame_info_dict['K'], pose_init[:3])
        #
        # kps_rep_error = np.linalg.norm(kpt2ds_f[kps2d_triang_ids] - rep3d, axis=1)
        # print(f'Triang points:'
        #       f'- min:{np.min(kps_rep_error)}\n'
        #       f'- max:{np.max(kps_rep_error)}\n'
        #       f'- med:{np.median(kps_rep_error)}')
        #
        # kps2d_exist_ids = np.where(kpt2d3d_ids_f[start_idx:] <= len(self.kpt3d_list))[0] + start_idx
        # kps2d_nonzero_ids = np.where(kpt2d3d_ids_f[start_idx:] >= 0)[0] + start_idx
        # kps2d_exist_ids = np.intersect1d(kps2d_exist_ids, kps2d_nonzero_ids)
        # kpt3d_exists_id = kpt2d3d_ids_f[kps2d_exist_ids]
        # kpt3d_exist = kpt3d_list_f[kpt3d_exists_id]
        # # rep3d_exist = project(kpt3d_exist, frame_info_dict['K'], frame_info_dict['pose_gt'][:3])
        # rep3d_exist = project(kpt3d_exist, frame_info_dict['K'], pose_init[:3])
        # kps_rep_error = np.linalg.norm(kpt2ds_f[kps2d_exist_ids] - rep3d_exist, axis=1)
        # print(f'Exist points:'
        #       f'- min:{np.min(kps_rep_error)}\n'
        #       f'- max:{np.max(kps_rep_error)}\n'
        #       f'- med:{np.median(kps_rep_error)}')
        #
        # from matplotlib import pyplot as plt
        # kps_disp = kpt2ds_f[kps2d_triang_ids]
        # Kps_rep = rep3d
        # plt.close()
        # plt.imshow(im_query)
        # plt.plot(Kps_rep[:, 0], Kps_rep[:, 1], 'bo')
        # plt.plot(kps_disp[:, 0], kps_disp[:, 1], 'r+')
        # plt.show()
        #
        # from matplotlib import pyplot as plt
        # kps_disp = kpt2ds_f[kps2d_exist_ids]
        # Kps_rep = rep3d_exist
        # plt.close()
        # plt.imshow(im_query)
        # plt.plot(Kps_rep[:, 0], Kps_rep[:, 1], 'bo')
        # plt.plot(kps_disp[:, 0], kps_disp[:, 1], 'r+')
        # plt.show()
        # ###################  Calculate Reprojection Error and visualization  ######################################

        # Apply BA with deep LM
        self.frame_visualization(kpt2ds_f, kpt2d3d_ids_f, kpt3d_list_f, cams_f)
        kpt3d_list_f, cams_f = self.apply_ba(kpt2ds_f, kpt2d3d_ids_f, kpt2d_fids_f, kpt3d_list_f, cams_f)
        self.frame_visualization(kpt2ds_f, kpt2d3d_ids_f, kpt3d_list_f, cams_f)
