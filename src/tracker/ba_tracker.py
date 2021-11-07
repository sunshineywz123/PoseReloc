import numpy as np
import cv2
from .vis_utils import Visualizer, project
from src.utils.vis_utils import reproj, ransac_PnP
from src.models.matchers.nn.nearest_neighbour import NearestNeighbour
device = 'cuda'
from matplotlib import pyplot as plt
import torch

def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
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
        self.kpt3d_sourceid = []
        self.kpt2d3d_ids = [] # 3D ids of each 2D keypoint
        self.update_th = 10
        self.frame_id = 0
        self.last_kf_info = None
        self.win_size = 20 # for static scene
        # self.win_size = 10 # for dynamic scene
        self.frame_interval = 3
        from src.utils.movie_writer import MovieWriter
        self.mw = MovieWriter()
        self.out = './track_kpt.mp4'
        self.init_cnt = 3
        self.use_motion_cnt = 0
        from src.tracker.vis_utils import Timer
        self.tm = Timer()

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
        if status is not None:
            valid_id = np.where(status.flatten() == 1)
            kpt_new = np.squeeze(kpt_new, axis=1)
            return kpt_new, valid_id
        else:
            return None, None

    def update_kf(self, kf_info_dict):
        if self.last_kf_info is not None:
            pose_last = self.pose_list[-1]
            trans_dist, rot_dist = self.cm_degree_5_metric(kf_info_dict['pose_pred'], pose_last)
            # [Option] Reject invalid updates
            if trans_dist > 10 or rot_dist > 10:
                trans_dist_gt, rot_dist_gt = self.cm_degree_5_metric(kf_info_dict['pose_pred'], kf_info_dict['pose_gt'])
                print(f"Update rejected:{trans_dist}/{rot_dist} - {trans_dist_gt}/{rot_dist_gt}")
                return False
            else:
                self.last_kf_info = kf_info_dict
                return True
            # self.last_kf_info = kf_info_dict
            # return True
        else:
            self.last_kf_info = kf_info_dict
            return True

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
            self.kpt3d_source_id = np.array(np.ones([len(self.kpt3d_list)], dtype=int) * self.id)
            self.kpt2d3d_ids = np.ones([n_kpt], dtype=int) * -1

            kf_3d_ids = np.arange(0, len(kf_info_dict['mkpts3d']))
            self.kpt2d3d_ids[kf_info_dict['valid_query_id']] = kf_3d_ids

            kf_db_ids = kf_info_dict['kpt3d_ids']
            # create mapping from DB id to kpt3d id
            self.db_3d_list = kf_db_ids
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
            # duplicate 3D keypoints in kf db_ids and current frmae
            intersect_kpts = np.intersect1d(kpt3d_db_ids, kf_db_ids)
            mask_kf_3d_exist = np.in1d(kf_db_ids, intersect_kpts) # [bool, ]
            mask_db3d_exist = np.in1d(kpt3d_db_ids, intersect_kpts) # [bool, ]

            # kf_3d_ids_ndup = np.where(mask_kf_3d_exist == False)[0] # non-duplicate kf 3d keypoint ids
            kf_kpt3ds_new = kf_info_dict['mkpts3d'][np.where(mask_kf_3d_exist == False)[0]] # get new 3D keypionts

            # Update 2D-3D relationship
            kf_kpt2d3d_id = np.ones([n_kpt]) * -1
            valid_2d_id = kf_info_dict['valid_query_id']

            # For duplicate parts, 3D ids copy from existing ids
            valid_id_dup = valid_2d_id[np.where(mask_kf_3d_exist == True)[0]]
            kf_kpt2d3d_id[valid_id_dup] = \
                np.arange(0, len(self.kpt3d_list))[np.where(mask_db3d_exist == True)[0]] # index on existing 3D db id

            # For non-duplicate parts, 3D ids are created
            kpt3d_start_id = len(self.kpt3d_list)
            valid_id_ndup = valid_2d_id[np.where(mask_kf_3d_exist == False)[0]]
            kf_kpt2d3d_id[valid_id_ndup] = np.arange(kpt3d_start_id, kpt3d_start_id + len(kf_kpt3ds_new))
            kf_kpt2d3d_id = np.asarray(kf_kpt2d3d_id, dtype=int)
            self.kpt2d3d_ids = np.concatenate([self.kpt2d3d_ids, kf_kpt2d3d_id], axis=0)

            # Update 3D keypoints
            self.kpt3d_list = np.concatenate([self.kpt3d_list, kf_kpt3ds_new], axis=0)
            self.kpt3d_source_id = np.concatenate([self.kpt3d_list, np.ones(kf_kpt3ds_new.shape[0],
                                                                            dtype=int) * self.id])

            # update mapping from DB id to kpt3d id
            kf_db_ids_new = kf_db_ids[valid_id_ndup] # non-duplicate kf 3d keypoint db id
            self.db_3d_list = np.concatenate([self.db_3d_list, kf_db_ids_new])
        # TODO: examine result after update

        self.last_kf_info = kf_info_dict
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
            print(f"pose_pred:{kf_pose_pred}")
            print(f"kpt3d_shape:{kf_kpts3d.shape} - {kf_kpts3d[:10]}")

            kf_kpts_proj = project(kf_kpts3d, K, kf_pose_pred[:3])

            # Load image and visualization
            if isinstance(kf_info['im_path'], str):
                kf_im_path = kf_info['im_path']
                kf_im = cv2.imread(kf_im_path)
            else:
                kf_im = kf_info['im_path']

            from matplotlib import pyplot as plt
            plt.close()
            plt.imshow(kf_im)
            plt.plot(kf_kps2d[:, 0], kf_kps2d[:, 1], 'r+')
            plt.plot(kf_kpts_proj[:, 0], kf_kpts_proj[:, 1], 'b+')
            plt.show()

    def cuda2cpu(self, pred_detection_cuda):
        return {k: v[0].cpu().numpy() for k, v in pred_detection_cuda.items()}

    def apply_match(self, kpt_pred0, kpt_pred1, T_0to1=None, K1=None, K2=None):
        import torch
        data = {}
        for k in kpt_pred0.keys():
            data[k + '0'] = kpt_pred0[k]
        for k in kpt_pred1.keys():
            data[k + '1'] = kpt_pred1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device) for k, v in data.items()}
        matching_result = self.matcher(data)

        # match01 = matching_result['matches0'].squeeze().cpu().numpy()
        # valid = np.where(match01 != -1)
        # matches_mask = np.copy(match01)
        # match01 = match01[valid[0]]
        # kps0 = kpt_pred0['keypoints'][valid[0]]
        # kps1 = kpt_pred1['keypoints'][match01]
        #
        # # ransac filtering
        # F, mask = cv2.findFundamentalMat(kps0, kps1, method=cv2.FM_RANSAC, ransacReprojThreshold=12.0)
        # if mask is not None:
        #     matchesMask = mask.ravel().tolist()
        #     for vid, kpt_id in enumerate(valid[0]):
        #         if matchesMask[vid] == 0:
        #             matches_mask[kpt_id] = - 1

        # if T_0to1 is not None:
        #     valid = np.where(matches_mask != -1)
        #     kpt_pred0['keypoints'][valid[0]]
        #     kpt_pred1['keypoints'][matches_mask]
        #     epipolar_error = compute_epipolar_error(kps0, kps1, T_0to1, K1, K2)
        #     epipolar_mask = epipolar_error < 5e-4
        #     for vid, kpt_id in enumerate(valid[0]):
        #         if epipolar_mask[vid] == False:
        #             matches_mask[kpt_id] = - 1

        # matching_result['matches0'] = torch.tensor(matches_mask).unsqueeze(0)
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

    def apply_triangulation(self, K1, K2, Tcw1, Tcw2, kpt2d_1, kpt2d_2):
        proj_mat1 = np.dot(K1, np.linalg.inv(Tcw1)[:3, :])
        proj_mat2 = np.dot(K2, np.linalg.inv(Tcw2)[:3, :])
        point_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, kpt2d_1.transpose(), kpt2d_2.transpose()).T
        point_3d_w = point_4d[:, :3] / np.repeat(point_4d[:, 3], 3).reshape(-1, 3)
        # point_3d_w2 = triangulatePy(proj_mat1, proj_mat2, kpt2d_1, kpt2d_2)
        return point_3d_w

    def motion_prediction(self):
        from transforms3d.euler import mat2euler, euler2mat
        pose0 = self.pose_list[-3]
        pose1 = self.pose_list[-2]
        pose_t = self.pose_list[-1]

        speed_trans = ((pose1[:3, 3] - pose0[:3, 3]) + (pose_t[:3, 3] - pose1[:3, 3]))/2

        rot0 = np.array(mat2euler(pose0[:3, :3]))
        rot1 = np.array(mat2euler(pose1[:3, :3]))
        rot_t = np.array(mat2euler(pose1[:3, :3]))
        speed_rot = ((rot1 - rot0) + (rot_t - rot1)) / 2
        trans_t = pose_t[:3, 3] + speed_trans
        rot_t = rot_t + speed_rot

        pose_new = np.eye(4)
        pose_new[:3, :3] = euler2mat(rot_t[0], rot_t[1], rot_t[2])
        pose_new[:3, 3] = trans_t
        return pose_new

    def flow_track(self, frame_info_dict, kf_frame_info):
        from src.tracker.vis_utils import put_text, draw_kpt2d
        if isinstance(frame_info_dict['im_path'], str):
            im_query_vis = cv2.imread(frame_info_dict['im_path'])
        else:
            im_query_vis = frame_info_dict['im_path']
        if isinstance(kf_frame_info['im_path'], str):
            im_kf = cv2.imread(kf_frame_info['im_path'], cv2.IMREAD_GRAYSCALE)
        else:
            im_kf = kf_frame_info['im_path']
            im_kf = cv2.cvtColor(im_kf, cv2.COLOR_BGR2GRAY)

        # Get initial pose with 2D-2D match from optical flow
        if isinstance(frame_info_dict['im_path'], str):
            im_query = cv2.imread(frame_info_dict['im_path'], cv2.IMREAD_GRAYSCALE)
        else:
            im_query = cv2.cvtColor(frame_info_dict['im_path'], cv2.COLOR_BGR2GRAY)

        mkpts2d_query, valid_ids = self.kpt_flow_track(im_kf, im_query, kf_frame_info['mkpts2d'])
        if valid_ids is not None:
            kpt3ds_kf = kf_frame_info['mkpts3d'][valid_ids]
            mkpts2d_query = mkpts2d_query[valid_ids]
            # Solve PnP to find initial pose
            pose_init, pose_init_homo, inliers = ransac_PnP(frame_info_dict['K_crop'], mkpts2d_query, kpt3ds_kf)
            kpt3d_rep = project(kpt3ds_kf, frame_info_dict['K_crop'], pose_init)
            kpt_dist = np.mean(np.linalg.norm(kpt3d_rep - mkpts2d_query, axis=1))

            trans_dist, rot_dist = self.cm_degree_5_metric(pose_init_homo, frame_info_dict['pose_gt'])

            print(f"\nFlow pose error:{trans_dist} - {rot_dist}\nKptDist:{kpt_dist}")
            label = f"Flow pose error:{trans_dist} - {rot_dist}"
            if trans_dist > 1:
                from matplotlib import pyplot as plt
                plt.imshow(im_query_vis)
                plt.plot(kpt3d_rep[:, 0], kpt3d_rep[:, 1], 'bo')
                plt.plot(mkpts2d_query[:, 0], mkpts2d_query[:, 1], 'r+')
                plt.savefig(f'./vis/{self.frame_id}.png')

            im_out = draw_kpt2d(im_query_vis, mkpts2d_query)
            im_out = put_text(im_out, label)
        else:
            im_out = im_query_vis
            kpt_dist = 1000086

        # scale = 1
        # h, w, c = im_out.shape
        # h_res = int(h / scale)
        # w_res = int(w / scale)
        # im_out = cv2.resize(im_out, (w_res, h_res))
        self.mw.write(im_out, self.out)
        if trans_dist <= 7 and rot_dist <= 7 and kpt_dist < 10:
            frame_info_dict['mkpts3d'] = kpt3ds_kf
            frame_info_dict['mkpts2d'] = mkpts2d_query
            self.last_kf_info = frame_info_dict

        # # Visualize correspondence
        # kpt2d_rep_q = project(kpt3ds_kf, frame_info_dict['K'],  frame_info_dict['pose_gt'][:3])
        # T_0to1 = np.dot(kf_frame_info['pose_gt'], np.linalg.inv(frame_info_dict['pose_gt']))
        # mkpt2ds_kf = kf_frame_info['mkpts2d'][valid_ids]
        # # kpt2d_rep_kf = project(kpt3ds, kf_frame_info['K'], kf_frame_info['pose_gt'][:3])
        # self.vis.add_kpt_corr(self.id, im_query, im_kf, mkpts2d_query, mkpt2ds_kf, kpt2d_proj=kpt2d_rep_q,
        #                       T_0to1=T_0to1, K=frame_info_dict['K'])
        return pose_init_homo, kpt_dist

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

        print(f'Input stat: {len(rep_error)}\n'
              f'- min:{np.min(rep_error)}\n'
              f'- max:{np.max(rep_error)}\n'
              f'- med:{np.median(rep_error)}\n'
              f'- mean:{np.mean(rep_error)}\n'
              f'- sum:{np.sum(rep_error)}')
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
            print(f'Frame:{frame_idx} - {len(kps_rep_error)}\n'
                  f'- min:{np.min(kps_rep_error)}\n'
                  f'- max:{np.max(kps_rep_error)}\n'
                  f'- med:{np.median(kps_rep_error)}\n'
                 f'- mean:{np.mean(kps_rep_error)}\n'
                  f'- sum:{np.sum(kps_rep_error)}')

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
        cams_K_np = cams[camera_idx, 6:]
        for frame_idx in np.unique(camera_idx):
            kpt_idx = np.where(camera_idx == frame_idx)[0]
            # print(len(kpt_idx))
            # print(kpt_idx[:10])
            # kpt_idx = kpt_idx[np.where(kpt3d_idx[kpt_idx] > len(self.kpt3d_list))]
            kps2d = kpt2ds_np[kpt_idx]
            kps3d = kpt3ds[kpt3d_idx[kpt_idx]]
            kps_cam = cams_np[kpt_idx]
            K, pose_mat = self.get_cam_params_back(kps_cam[0])
            kps_rep = project(kps3d, K, pose_mat[:3])
            kps_rep_error = np.linalg.norm(kps2d - kps_rep, axis=1)
            print(f'Frame:{frame_idx} - {len(kps_rep_error)}\n'
                  f'- min:{np.min(kps_rep_error)}\n'
                  f'- max:{np.max(kps_rep_error)}\n'
                  f'- med:{np.median(kps_rep_error)}\n'
                  f'- mean:{np.mean(kps_rep_error)}\n'
                  f'- sum:{np.sum(kps_rep_error)}')
        # points_opt_np = points.cpu().numpy()
        # cam_opt_np = cameras.cpu().numpy()
        cam_opt = np.concatenate([cam_opt_np, cams[:, 6:]], axis=1)
        return points_opt_np, cam_opt

    def apply_ba_V2(self, kpt2ds, kpt2d3d_ids, kpt2d_fids, kpt3d_list, cams, verbose=False):
        from DeepLM.BAProblem.loss import SnavelyReprojectionError, SnavelyReprojectionErrorV2
        from DeepLM.TorchLM.solver import Solve
        import torch
        device = 'cuda'
        points = torch.tensor(kpt3d_list, device=device, dtype=torch.float64, requires_grad=False)
        cam_pose = torch.tensor(cams[:, :6], device=device, dtype=torch.float64, requires_grad=False)
        valid2d_idx = np.where(kpt2d3d_ids != -1)[0]
        if np.max(kpt2d_fids) > self.win_size:
            print("[START FILTERING +++++++++++++++]")
            valid_idx_fid = np.where(kpt2d_fids > np.max(kpt2d_fids) - self.win_size)
            valid2d_idx = np.intersect1d(valid_idx_fid, valid2d_idx)
        ks = cams[np.array(kpt2d_fids[valid2d_idx], dtype=int), 6:]
        features = torch.tensor(np.concatenate([kpt2ds[valid2d_idx], ks], axis=1), device=device, dtype=torch.float64,
                                requires_grad=False)
        ptIdx = torch.tensor(kpt2d3d_ids[valid2d_idx], device=device, dtype=torch.int64, requires_grad=False)
        camIdx = torch.tensor(kpt2d_fids[valid2d_idx], device=device, dtype=torch.int64, requires_grad=False)
        print(f"Num BA pts:{len(features)}")

        if verbose:
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
                print(f'Frame:{frame_idx} with {len(kps2d)} kpts\n'
                      f'- min:{np.min(kps_rep_error)}\n'
                      f'- max:{np.max(kps_rep_error)}\n'
                      f'- med:{np.median(kps_rep_error)}\n'
                      f'- sum:{np.sum(kps_rep_error)}')
        points, cam_pose, features, ptIdx, camIdx = points.to(device), \
                                                   cam_pose.to(device), features.to(device), \
                                                    ptIdx.to(device), camIdx.to(device)

        if device == 'cuda':
            torch.cuda.synchronize()

        # optimize
        Solve(variables=[points, cam_pose],
              constants=[features],
              indices=[ptIdx, camIdx],
              fn=SnavelyReprojectionErrorV2,
              numIterations=5,
              numSuccessIterations=5,
              verbose=verbose)

        points_opt_np = points.cpu().detach().numpy()
        cam_opt_np = cam_pose.cpu().detach().numpy()

        # Display Optimized Reprojection Error by frame
        if verbose:
            kpt2ds_np = kpt2ds[valid2d_idx]
            kpt3d_idx = kpt2d3d_ids[valid2d_idx]
            camera_idx = np.asarray(kpt2d_fids[valid2d_idx], dtype=int)
            kpt3ds = points_opt_np
            cams_np = cam_opt_np[camera_idx]
            cams_K_np = cams[camera_idx, 6:]
            for frame_idx in np.unique(camera_idx):
                kpt_idx = np.where(camera_idx == frame_idx)[0]
                # print(len(kpt_idx))
                # print(kpt_idx[:10])
                # kpt_idx = kpt_idx[np.where(kpt3d_idx[kpt_idx] > len(self.kpt3d_list))]
                kps2d = kpt2ds_np[kpt_idx]
                kps3d = kpt3ds[kpt3d_idx[kpt_idx]]
                kps_cam = cams_np[kpt_idx]
                kps_cam_K = cams_K_np[kpt_idx]
                kps_cam_input = np.concatenate([kps_cam[0], kps_cam_K[0]])
                K, pose_mat = self.get_cam_params_back(kps_cam_input)
                kps_rep = project(kps3d, K, pose_mat[:3])
                kps_rep_error = np.linalg.norm(kps2d - kps_rep, axis=1)
                print(f'Frame:{frame_idx}\n'
                      f'- min:{np.min(kps_rep_error)}\n'
                      f'- max:{np.max(kps_rep_error)}\n'
                      f'- med:{np.median(kps_rep_error)}\n'
                      f'- sum:{np.sum(kps_rep_error)}')
        # points_opt_np = points.cpu().numpy()
        # cam_opt_np = cameras.cpu().numpy()
        cam_opt = np.concatenate([cam_opt_np, cams[:, 6:]], axis=1)
        return points_opt_np, cam_opt

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

    def track_ba(self, frame_info_dict, verbose=True):
        import  torch
        print(f"Updating frame id:{self.frame_id} [WIN SIZE:{len(np.unique(self.kpt2d_fids))}]")
        ba_log = dict()

        pose_init = frame_info_dict['pose_init']
        # Load image
        kf_frame_info = self.kf_frames[self.last_kf_id]
        if isinstance(kf_frame_info['im_path'], str):
            im_kf = cv2.imread(kf_frame_info['im_path'], cv2.IMREAD_GRAYSCALE)
        else:
            im_kf = cv2.cvtColor(kf_frame_info['im_path'], cv2.COLOR_BGR2GRAY)

        # self.frame_visualization()

        # [Option] Extract and match 2D keypoints from original image
        # im_query = cv2.imread(frame_info_dict['im_path_orig'], cv2.IMREAD_GRAYSCALE)
        # im_query = im_query.astype(np.float32)
        # im_query = im_query[None]
        # im_query /= 255.
        # inp = torch.tensor(im_query, device='cuda')
        # inp = inp.unsqueeze(0)
        # with torch.no_grad():
        #     kpt2ds_pred_query = self.cuda2cpu(self.extractor(inp))

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
        T_0to1 = np.dot(np.linalg.inv(kf_frame_info['pose_gt']), frame_info_dict['pose_gt'])
        print(f"Input kf:{len(kpt2ds_pred_kf['keypoints'])} - {len(kpt2ds_pred_query['keypoints'])}")
        match_results = self.apply_match(kpt2ds_pred_kf, kpt2ds_pred_query, T_0to1=T_0to1,
                                         K1=kf_frame_info['K'], K2=frame_info_dict['K'])

        match_kq = match_results['matches0'][0].cpu().numpy()
        valid = np.where(match_kq != -1)
        mkpts2d_kf = kpt2ds_pred_kf['keypoints'][valid]
        mkpts2d_query = kpt2ds_pred_query['keypoints'][match_kq[valid]]
        kpt_idx_valid = kpt_idx[valid]

        self.vis.set_new_seq('match_res_40')
        if isinstance(frame_info_dict['im_path'], str):
            im_query = cv2.imread(frame_info_dict['im_path'])
        else:
            im_query = frame_info_dict['im_path']
        self.vis.add_kpt_corr(self.frame_id, im_kf, im_query, mkpts2d_kf, mkpts2d_query)

        # Update
        # kpt2ds_match_f = np.copy(self.kpt2d_match)
        kpt2d3d_ids_f = np.copy(self.kpt2d3d_ids)

        # Update 2D inform
        n_kpt_q = len(mkpts2d_query)
        kpt2ds_f = np.concatenate([self.kpt2ds, mkpts2d_query]) # update 2D keypoints
        # kpt2ds_match_f[valid] += 1 # update 2D match
        # kpt2ds_match_f = np.concatenate([kpt2ds_match_f, np.ones([n_kpt_q])])

        # Check 2D-3D correspondence
        kf_2d_3d_ids = kpt2d3d_ids_f[kpt_idx_valid]
        kpt_idx_wo3d = np.where(kf_2d_3d_ids == -1)[0] # local index of point without 3D index
        mkpts2d_kf_triang = mkpts2d_kf[kpt_idx_wo3d]
        mkpts2d_query_triang = mkpts2d_query[kpt_idx_wo3d]

        # Update 2D-3D correspondence for existing points
        query_2d3d_ids = np.ones(n_kpt_q) * -1
        # estimate reprojection error
        kpt_idx_w3d = np.where(kf_2d_3d_ids != -1)[0]
        print(f"Found {len(kpt_idx_w3d)} points with known 3D")
        ba_log['pt_found'] = len(kpt_idx_w3d)

        mkps3d_query_exist = self.kpt3d_list[kf_2d_3d_ids[kpt_idx_w3d]]
        mkpts2d_query_exist = mkpts2d_query[kpt_idx_w3d]
        kpt2d_rep_query_exist = project(mkps3d_query_exist, frame_info_dict['K'], pose_init[:3])
        rep_diff_q_exist = np.linalg.norm(kpt2d_rep_query_exist - mkpts2d_query_exist, axis=1)

        # VISUALIZATION: Examine source of kpt3d used
        kpt3d_source_ids = self.kpt3d_source_id[np.array(kf_2d_3d_ids[kpt_idx_w3d], dtype=int)]
        unique, counts = np.unique(kpt3d_source_ids, return_counts=True)
        kpt3s_stat_init = dict(zip(unique, counts))

        # [Option] keep only points with reprojection error smaller than 20
        med = np.median(rep_diff_q_exist) * 1.2
        exist_keep_idx_q_rep = np.where(rep_diff_q_exist < med)[0]
        from matplotlib import pyplot as plt
        plt.close()
        # mean = np.mean(rep_diff_q_exist)
        # med = np.median(rep_diff_q_exist)
        label = f'med:{med}'
        plt.title(label)
        plt.hist(rep_diff_q_exist, histtype='bar', align='mid')
        plt.savefig(f'./vis/repdist_stat/{self.frame_id}.png'.zfill(5))

        # [Option] only remove points from rescently initialized frames
        # rep_source_ids = self.kpt3d_source_id[kf_2d_3d_ids[kpt_idx_w3d]]
        # exist_keep_idx_q_source = np.where(self.id - rep_source_ids > 5)[0]
        # exist_keep_idx_q = np.unique(np.concatenate([exist_keep_idx_q_source, exist_keep_idx_q_rep]))

        exist_keep_idx_q = exist_keep_idx_q_rep
        kpt_idx_w3d_keep = kpt_idx_w3d[exist_keep_idx_q]
        print(f"Known point removed:{len(np.where(rep_diff_q_exist >= 20)[0])}")

        # kpt_idx_w3d_keep = kpt_idx_w3d
        query_2d3d_ids[kpt_idx_w3d_keep] = kf_2d_3d_ids[kpt_idx_w3d_keep]

        # VISUALIZATION: Examine source of kpt3d used
        assert len(self.kpt3d_source_id) == len(self.kpt3d_list), "Length non-equal"
        kpt3d_source_ids = self.kpt3d_source_id[np.array(query_2d3d_ids[np.where(query_2d3d_ids != -1)], dtype=int)]
        unique, counts = np.unique(kpt3d_source_ids, return_counts=True)
        kpt3s_stat_final = dict(zip(unique, counts))
        print(f"Kpt3d -{self.id}:\n{kpt3s_stat_init}\n{kpt3s_stat_final}")

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

        # Triangulation
        if len(kpt_idx_wo3d) > 0:
            Tco_kf = np.linalg.inv(kf_frame_info['pose_pred'])
            Tco_query = np.linalg.inv(pose_init)
            kpt3ds_triang = self.apply_triangulation(kf_frame_info['K'],
                                                     frame_info_dict['K'],
                                                     Tco_kf, Tco_query,
                                                     mkpts2d_kf_triang, mkpts2d_query_triang)

            # Remove triangulation points with extremely large error
            kpt2d_rep_kf = project(kpt3ds_triang, kf_frame_info['K'],  kf_frame_info['pose_pred'][:3])
            kpt2d_rep_query = project(kpt3ds_triang, frame_info_dict['K'],  pose_init[:3])
            rep_diff_q = np.linalg.norm(kpt2d_rep_query - mkpts2d_query_triang, axis=1)
            rep_diff_kf = np.linalg.norm(kpt2d_rep_kf - mkpts2d_kf_triang, axis=1)
            # Remove 3D points with large error
            triang_rm_idx_q = np.where(rep_diff_q > 20)[0]
            triang_rm_idx_kf = np.where(rep_diff_kf > 20)[0]

            # [Option] Remove 3D points distant away
            triang_rm_idx_dist = np.where(kpt3ds_triang[:, 2] > 0.15)[0]
            triang_rm_idx = np.unique(np.concatenate([triang_rm_idx_q, triang_rm_idx_kf, triang_rm_idx_dist]))
            # triang_rm_idx = np.unique(np.concatenate([triang_rm_idx_q, triang_rm_idx_kf]))

            triang_keep_idx = np.array([i for i in range(len(kpt2d_rep_query))
                                        if i not in triang_rm_idx]) # index over mkpts2d_q_triang

            print(f"{len(triang_rm_idx)} removed out of {len(kpt_idx_wo3d)} points")
            ba_log['pt_triang'] = len(kpt_idx_wo3d)
            ba_log['pt_triang_rm'] = len(triang_rm_idx)

            if len(triang_keep_idx) != 0:
                mkpts2d_kf_triang = mkpts2d_kf_triang[triang_keep_idx]
                mkpts2d_query_triang = mkpts2d_query_triang[triang_keep_idx]
                kpt2d_rep_kf = kpt2d_rep_kf[triang_keep_idx]
                kpt2d_rep_query = kpt2d_rep_query[triang_keep_idx]
        else:
            triang_keep_idx = []

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

        # Update correspondence for newly triangulated points
        if len(kpt_idx_wo3d) > 0 and len(triang_keep_idx) > 0:
            kpt3d_start_id = len(self.kpt3d_list)
            query_2d3d_ids[kpt_idx_wo3d[triang_keep_idx]] = np.arange(kpt3d_start_id, kpt3d_start_id
                                                                      + len(triang_keep_idx))

        query_2d3d_ids = np.asarray(query_2d3d_ids, dtype=int)
        kpt2d3d_ids_f = np.concatenate([self.kpt2d3d_ids, query_2d3d_ids])

        # Add 3D points
        if len(kpt_idx_wo3d) > 0 and len(triang_keep_idx) > 0:
            kpt3d_list_f = np.concatenate([self.kpt3d_list, kpt3ds_triang[triang_keep_idx]])
        else:
            kpt3d_list_f = np.copy(self.kpt3d_list)
        cams_f = np.concatenate([self.cams,  [self.get_cam_param(frame_info_dict['K'], pose_init)]])
        kpt2d_fids_f = np.concatenate([self.kpt2d_fids, np.ones([n_kpt_q]) * self.id])

        n_triang_pt = len(triang_keep_idx)
        if verbose:
            # ###################  Calculate Reprojection Error and visualization  ############################
            kpt_idxs = np.where(kpt2d_fids_f == np.max(kpt2d_fids_f))[0]
            start_idx = np.min(kpt_idxs)
            kpt_idxs = kpt_idxs[np.where(kpt2d3d_ids_f[kpt_idxs] != -1)[0]]
            if len(kpt_idxs) != 0:
                # print(len(kpt_idxs))
                # print(kpt_idxs[:10])
                kpt3d_full = kpt3d_list_f[kpt2d3d_ids_f[kpt_idxs]]
                kpt2d_full = kpt2ds_f[kpt_idxs]
                # print(kpt2d3d_ids_f[kpt_idxs][:10])
                # rep3d_full = project(kpt3d_full, frame_info_dict['K'], frame_info_dict['pose_gt'][:3])
                rep3d_full = project(kpt3d_full, frame_info_dict['K'], pose_init[:3])
                kps_error_full = np.linalg.norm(kpt2d_full- rep3d_full, axis=1)
                print(f'Full points: {len(kps_error_full)}'
                      f'- min:{np.min(kps_error_full)}\n'
                      f'- max:{np.max(kps_error_full)}\n'
                      f'- med:{np.median(kps_error_full)}')

            if n_triang_pt > 0:
                kps2d_triang_ids = np.where(kpt2d3d_ids_f[start_idx:] >= len(self.kpt3d_list))[0] + start_idx
                kpt3d_triang_ids = kpt2d3d_ids_f[kps2d_triang_ids]
                rep3d = project(kpt3d_list_f[kpt3d_triang_ids], frame_info_dict['K'], pose_init[:3])
                kps_rep_error = np.linalg.norm(kpt2ds_f[kps2d_triang_ids] - rep3d, axis=1)
                print(f'Triang points:{len(kps_rep_error)}'
                      f'- min:{np.min(kps_rep_error)}\n'
                      f'- max:{np.max(kps_rep_error)}\n'
                      f'- med:{np.median(kps_rep_error)}')

            kps2d_exist_ids = np.where(kpt2d3d_ids_f[start_idx:] < len(self.kpt3d_list))[0] + start_idx
            if len(kps2d_exist_ids) > 0:
                kps2d_nonzero_ids = np.where(kpt2d3d_ids_f[start_idx:] >= 0)[0] + start_idx
                kps2d_exist_ids = np.intersect1d(kps2d_exist_ids, kps2d_nonzero_ids)
                if len(kps2d_exist_ids) != 0:
                    # FIXME: might be problem here
                    kpt3d_exists_id = kpt2d3d_ids_f[kps2d_exist_ids]
                    kpt3d_exist = kpt3d_list_f[kpt3d_exists_id]
                    # rep3d_exist = project(kpt3d_exist, frame_info_dict['K'], frame_info_dict['pose_gt'][:3])
                    rep3d_exist = project(kpt3d_exist, frame_info_dict['K'], pose_init[:3])
                    kps_rep_error = np.linalg.norm(kpt2ds_f[kps2d_exist_ids] - rep3d_exist, axis=1)
                    print(f'Exist points: - {len(kps_rep_error)}\n'
                          f'- min:{np.min(kps_rep_error)}\n'
                          f'- max:{np.max(kps_rep_error)}\n'
                          f'- med:{np.median(kps_rep_error)}')

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
        # self.frame_visualization(kpt2ds_f, kpt2d3d_ids_f, kpt3d_list_f, cams_f)
        # kpt3d_list_f, cams_f = self.apply_ba(kpt2ds_f, kpt2d3d_ids_f, kpt2d_fids_f, kpt3d_list_f, cams_f)
        kpt3d_list_f, cams_f = self.apply_ba_V2(kpt2ds_f, kpt2d3d_ids_f, kpt2d_fids_f,
                                                kpt3d_list_f, cams_f, verbose=False)

        # self.frame_visualization(kpt2ds_f, kpt2d3d_ids_f, kpt3d_list_f, cams_f)
        K_opt, pose_opt = self.get_cam_params_back(cams_f[-1])

        trans_dist_pred, rot_dist_pred = self.cm_degree_5_metric(frame_info_dict['pose_pred'], frame_info_dict['pose_gt'])
        trans_dist_pred = np.round(trans_dist_pred, decimals=2)
        rot_dist_pred = np.round(rot_dist_pred, decimals=2)
        ba_log['pred_err_trans'] = trans_dist_pred
        ba_log['pred_err_rot'] = rot_dist_pred
        print(f"Pred pose error:{ba_log['pred_err_trans']} - {ba_log['pred_err_rot']}")

        trans_dist_init, rot_dist_init = self.cm_degree_5_metric(pose_init, frame_info_dict['pose_gt'])
        trans_dist_init = np.round(trans_dist_init, decimals=2)
        rot_dist_init = np.round(rot_dist_init, decimals=2)
        ba_log['init_err_trans'] = trans_dist_init
        ba_log['init_err_rot'] = rot_dist_init
        print(f"Initial pose error:{ba_log['init_err_trans']} - {ba_log['init_err_rot']}")

        trans_dist, rot_dist = self.cm_degree_5_metric(pose_opt, frame_info_dict['pose_gt'])
        trans_dist = np.round(trans_dist, decimals=2)
        rot_dist = np.round(rot_dist, decimals=2)
        ba_log['opt_err_trans'] = trans_dist
        ba_log['opt_err_rot'] = rot_dist
        print(f"Optimized pose error:{ba_log['opt_err_trans']} - {ba_log['opt_err_rot']}")

        # trans_improv = np.abs(trans_dist - trans_dist_init)
        # rot_improv = np.abs(rot_dist - rot_dist_init)
        # update_valid = trans_improv < 3 and rot_improv < 3
        update_valid = True
        if False:
            # Compute and visualize final error
            # Get 2D and 3D points of current frame
            kpt_idxs = np.where(kpt2d_fids_f == np.max(kpt2d_fids_f))[0]
            kpt_idxs = kpt_idxs[np.where(kpt2d3d_ids_f[kpt_idxs] != -1)[0]]
            # print(len(kpt_idxs))
            # print(kpt_idxs[:10])
            # print(kpt2d3d_ids_f[kpt_idxs][:10])

            frame_info_dict['mkpts2d'] = kpt2ds_f[kpt_idxs]
            frame_info_dict['mkpts3d'] = kpt3d_list_f[kpt2d3d_ids_f[kpt_idxs]]
            kpt3d_rep = project(frame_info_dict['mkpts3d'], K_opt, pose_opt[:3])
            rep_err = np.linalg.norm(kpt3d_rep - frame_info_dict['mkpts2d'], axis=1)
            kpt3d_rep2 = project(frame_info_dict['mkpts3d'], frame_info_dict['K'], frame_info_dict['pose_gt'][:3])
            rep_err2 = np.linalg.norm(kpt3d_rep2 - frame_info_dict['mkpts2d'], axis=1)
            print(f"K optimized:{frame_info_dict['K']  - K_opt}")
            frame_info_dict['K'] = K_opt
            print(f'KF Update error opt: {len(kps_rep_error)}\n'
                  f'- min:{np.min(rep_err)}\n'
                  f'- max:{np.max(rep_err)}\n'
                  f'- med:{np.median(rep_err)}')
            print(f'KF Update error2: {len(rep_err2)}\n'
                  f'- min:{np.min(rep_err2)}\n'
                  f'- max:{np.max(rep_err2)}\n'
                  f'- med:{np.median(rep_err2)}')

            kpt3d_rep2 = kpt3d_rep2[np.where(rep_err2 <= 50)[0]]
            from matplotlib import pyplot as plt
            im_query = cv2.imread(frame_info_dict['im_path'], cv2.IMREAD_GRAYSCALE)
            plt.close()
            plt.imshow(im_query)
            plt.plot(kpt3d_rep[:, 0], kpt3d_rep[:, 1], 'r+')
            plt.show()
            plt.close()
            plt.imshow(im_query)
            plt.plot(kpt3d_rep2[:, 0], kpt3d_rep2[:, 1], 'r+')
            plt.show()
        # if len(triang_keep_idx) > self.update_th:
        if (self.frame_id % self.frame_interval == 0 and update_valid) or self.init_cnt > 0 :
            print(f"Num updated :{len(triang_keep_idx)}")

            unmatched_idx = np.array([i for i in range(len(kpt2ds_pred_query['keypoints'])) if i not in match_kq])
            num_unmatch = len(unmatched_idx)
            self.kpt2ds = np.concatenate([kpt2ds_f, kpt2ds_pred_query['keypoints'][unmatched_idx]])
            self.kpt2d_descs = np.concatenate([self.kpt2d_descs,
                                               kpt2ds_pred_query['descriptors'][:, match_kq[valid]].transpose(),
                                               kpt2ds_pred_query['descriptors'][:, unmatched_idx].transpose()])

            self.kpt2d_fids = np.concatenate([kpt2d_fids_f, np.ones(num_unmatch, dtype=int) * kpt2d_fids_f[-1]])
            self.cams = cams_f
            # self.kpt2ds_match = kpt2ds_match_f
            self.kpt3d_source_id = np.concatenate([self.kpt3d_source_id, np.ones([len(kpt3d_list_f) - len(self.kpt3d_list)],
                                                                            dtype=int) * self.id])
            self.kpt3d_list = kpt3d_list_f
            self.kpt2d3d_ids = np.concatenate([kpt2d3d_ids_f, np.ones(num_unmatch, dtype=int) * -1])
            frame_info_dict['pose_pred'] = pose_init

            # kpt_idxs_full = np.where(kpt2d_fids_f == np.max(kpt2d_fids_f))[0]
            # kpt_idxs_w3d = kpt_idxs_full[np.where(kpt2d3d_ids_f[kpt_idxs_full] != -1)[0]]

            self.kf_kpt_index_dict[self.id] = (len(self.kpt2d_fids) - 1 - len(kpt2ds_pred_query['keypoints']),
                                               len(self.kpt2d_fids)-1)
            self.kf_frames.pop(self.last_kf_id)
            self.kf_frames[self.id] = frame_info_dict

            self.last_kf_id = self.id
            self.id += 1

            if np.max(self.kpt2d_fids) > self.win_size:
                print("[START Deleting frame infos]")

                valid_idx_fid = np.where(self.kpt2d_fids > np.max(self.kpt2d_fids) - self.win_size)
                num_del = len(self.kpt2d_fids) - len(valid_idx_fid)
                start_idx, end_idx = self.kf_kpt_index_dict[self.last_kf_id]
                self.kf_kpt_index_dict[self.last_kf_id] = [start_idx - num_del, end_idx - num_del]
                self.kpt2d_fids = self.kpt2d_fids[valid_idx_fid]
                self.kpt2ds = self.kpt2ds[valid_idx_fid]
                self.kpt2d3d_ids = self.kpt2d3d_ids[valid_idx_fid]
                self.kpt2d_descs = self.kpt2d_descs[valid_idx_fid]

        self.frame_id += 1
        return pose_opt, ba_log, update_valid

    def track(self, frame_info_dict, flow_track_only=False, auto_mode=False):
        if not auto_mode:
            # self.init_cnt = -1
            pose_ftk, kpt_dist = self.flow_track(frame_info_dict, self.last_kf_info)
            # pose_ftk = frame_info_dict['pose_pred']

            trans_dist_fkt, rot_dist_fkt = self.cm_degree_5_metric(self.pose_list[-1], pose_ftk)
            trans_dist_gt_fkt, rot_dist_gt_fkt = self.cm_degree_5_metric(frame_info_dict['pose_gt'], pose_ftk)

            # [Option] Whether or not to use Full Image
            # import os.path as osp
            # img_path = frame_info_dict['im_path']
            # im_dir = osp.dirname(img_path.replace('color/', 'rgb/'))
            # im_name = 'image' + osp.basename(img_path).replace('.png', '') + '.png'
            # frame_info_dict['im_path_orig'] = osp.join(im_dir, im_name)

            if len(self.pose_list) < 3:
                pose_mo = frame_info_dict['pose_pred']
            else:
                pose_mo = self.motion_prediction()

            trans_dist_gt_fkt, rot_dist_gt_fkt = self.cm_degree_5_metric(frame_info_dict['pose_gt'], pose_ftk)
            print(f"Flow dist:{trans_dist_fkt}/{rot_dist_fkt} - {trans_dist_gt_fkt}/{rot_dist_gt_fkt}")
            trans_dist_gt_mo, rot_dist_gt_mo = self.cm_degree_5_metric(frame_info_dict['pose_gt'], pose_mo)
            print(f"MP dist:{trans_dist_gt_mo}/{rot_dist_gt_mo}")

            # [Option] use motion prediction when keypoint tracking is not valid
            if ((trans_dist_fkt > 10 or rot_dist_fkt > 10) and self.use_motion_cnt < 3) or kpt_dist > 10:
                print(f"+++++++++++ Using motion model")
                frame_info_dict['pose_init'] = pose_mo
                self.use_motion_cnt += 1
            elif (trans_dist_fkt < 20 and rot_dist_fkt < 20):
                frame_info_dict['pose_init'] = pose_ftk
                self.use_motion_cnt = 0
            else:
                print("+++++++++++ Using motion model F")
                frame_info_dict['pose_init'] = pose_mo
                self.use_motion_cnt += 1

            # initialize with gt
            if self.init_cnt > 0:
                print("======= INITIALIZING")
                self.init_cnt -= 1
                frame_info_dict['pose_init'] = frame_info_dict['pose_gt']

        else:
            if self.init_cnt > 0:
                print("======= INITIALIZING")
                self.init_cnt -= 1
                frame_info_dict['pose_init'] = frame_info_dict['pose_gt']
            else:
                pose_mo = self.motion_prediction()
                frame_info_dict['pose_init'] = pose_mo

        if not flow_track_only:
            pose_opt, ba_log, update_valid = self.track_ba(frame_info_dict, verbose=False)
            self.pose_list.append(pose_opt)
        else:
            pose_init = frame_info_dict['pose_init']
            pose_opt = frame_info_dict['pose_init']
            ba_log = dict()
            trans_dist_pred, rot_dist_pred = self.cm_degree_5_metric(frame_info_dict['pose_pred'],
                                                                     frame_info_dict['pose_gt'])
            trans_dist_pred = np.round(trans_dist_pred, decimals=2)
            rot_dist_pred = np.round(rot_dist_pred, decimals=2)
            ba_log['pred_err_trans'] = trans_dist_pred
            ba_log['pred_err_rot'] = rot_dist_pred
            print(f"Pred pose error:{ba_log['pred_err_trans']} - {ba_log['pred_err_rot']}")

            trans_dist_init, rot_dist_init = self.cm_degree_5_metric(pose_init, frame_info_dict['pose_gt'])
            trans_dist_init = np.round(trans_dist_init, decimals=2)
            rot_dist_init = np.round(rot_dist_init, decimals=2)
            ba_log['init_err_trans'] = trans_dist_init
            ba_log['init_err_rot'] = rot_dist_init
            print(f"Initial pose error:{ba_log['init_err_trans']} - {ba_log['init_err_rot']}")

            trans_dist, rot_dist = self.cm_degree_5_metric(pose_opt, frame_info_dict['pose_gt'])
            trans_dist = np.round(trans_dist, decimals=2)
            rot_dist = np.round(rot_dist, decimals=2)
            ba_log['opt_err_trans'] = trans_dist
            ba_log['opt_err_rot'] = rot_dist
            print(f"Optimized pose error:{ba_log['opt_err_trans']} - {ba_log['opt_err_rot']}")
            self.pose_list.append(frame_info_dict['pose_init'])

        # if trans_dist > 5:
        #     pose_opt = self.track_ba(frame_info_dict)
        return frame_info_dict['pose_init'], pose_opt, ba_log