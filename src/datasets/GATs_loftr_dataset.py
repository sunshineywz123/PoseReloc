from loguru import logger

try:
    import ujson as json
except ImportError:
    import json
import torch
import numpy as np
import time

from pycocotools.coco import COCO
from torch.utils.data import Dataset
import open3d as o3d
from scipy import stats
from .utils import read_grayscale
from src.utils import data_utils


class GATsLoFTRDataset(Dataset):
    def __init__(
        self,
        anno_file,
        num_leaf,
        pad=True,
        img_pad=False,
        img_resize=512,
        coarse_scale=1 / 8,
        df=8,
        shape2d=2000,
        shape3d=10000,
        percent=1.0,
        split="train",
        load_pose_gt=False,
        path_prefix_substitute_3D_source=None,
        path_prefix_substitute_3D_aim=None,
        path_prefix_substitute_2D_source=None,
        path_prefix_substitute_2D_aim=None,
        downsample=False,
        downsample_resolution=30
    ):
        super(Dataset, self).__init__()

        self.split = split
        self.coco = COCO(anno_file)
        self.anns = np.array(self.coco.getImgIds())

        logger.info(f"Use {percent * 100}% data")
        sample_inverval = int(1 / percent)
        self.anns = self.anns[::sample_inverval]

        self.num_leaf = num_leaf
        self.load_pose_gt = load_pose_gt

        # 3D point cloud part
        self.pad = pad
        self.shape2d = shape2d
        self.shape3d = shape3d

        # 2D query image part
        self.img_pad = img_pad
        self.img_resize = img_resize
        self.df = df
        self.coarse_scale = coarse_scale

        # Downsample
        self.downsample = downsample
        self.downsample_resolution = downsample_resolution

        # For data path substiture to use data generated at other clusters
        self.path_prefix_substitute_3D_source = (str(path_prefix_substitute_3D_source),)
        self.path_prefix_substitute_3D_aim = (str(path_prefix_substitute_3D_aim),)
        self.path_prefix_substitute_2D_source = (str(path_prefix_substitute_2D_source),)
        self.path_prefix_substitute_2D_aim = (str(path_prefix_substitute_2D_aim),)

    def read_anno2d(self, anno2d_file):
        """ Read (and pad) 2d info"""
        with open(anno2d_file, "r") as f:
            data = json.load(f)

        keypoints2d = torch.Tensor(data["keypoints2d"])  # [n, 2]
        scores2d = torch.Tensor(data["scores2d"])  # [n, 1]
        assign_matrix = torch.Tensor(data["assign_matrix"])  # [2, k]

        num_2d_orig = keypoints2d.shape[0]

        return keypoints2d, scores2d, assign_matrix, num_2d_orig

    def voxel_filter(pts, scores, grid_size, use_3d=True, min_num_pts=4):
        mins = pts.min(axis=0) - grid_size
        maxs = pts.max(axis=0) + grid_size
        bins = [np.arange(mins[i], maxs[i], grid_size) for i in range(len(mins))]

        si = 2
        if use_3d:
            si = 3

        counts, edges, binnumbers = stats.binned_statistic_dd(
            pts[:, :si],
            values=None,
            statistic="count",
            bins=bins[:si],
            range=None,
            expand_binnumbers=False,
        )

        ub = np.unique(binnumbers)
        pts_ds = []
        scores_ds = []
        for b in ub:
            if len(np.where(binnumbers == b)[0]) >= min_num_pts:
                pts_ds.append(pts[np.where(binnumbers == b)[0]].mean(axis=0))
                scores_ds.append(scores[np.where(binnumbers == b)[0]].mean())
        pts_ds = np.vstack(pts_ds)
        scores_ds = np.vstack(scores_ds).reshape(-1)
        return pts_ds, scores_ds

    def read_anno3d(
        self, avg_anno3d_file, clt_anno3d_file, idxs_file, pad=True, assignmatrix=None
    ):
        """ Read(and pad) 3d info"""
        avg_data = np.load(avg_anno3d_file)
        # with open(avg_anno3d_file, 'r') as f:
        # avg_data = json.load(f)

        clt_data = np.load(clt_anno3d_file)
        # with open(collect_anno3d_file, 'r') as f:
        # collect_data = json.load(f)

        idxs = np.load(idxs_file)

        keypoints3d = torch.Tensor(clt_data["keypoints3d"])  # [m, 3]
        avg_descriptors3d = torch.Tensor(avg_data["descriptors3d"])  # [dim, m]
        clt_descriptors = torch.Tensor(clt_data["descriptors3d"])  # [dim, k]
        avg_scores = torch.Tensor(avg_data["scores3d"])  # [m, 1]
        clt_scores = torch.Tensor(clt_data["scores3d"])  # [k, 1]


        clt_descriptors, clt_scores = data_utils.build_features3d_leaves(
            clt_descriptors, clt_scores, idxs, num_leaf=self.num_leaf
        )

        # Pointcloud downsampling use by voxel
        if self.downsample:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(keypoints3d.numpy())
            bbox_size = pcd.get_max_bound() - pcd.get_min_bound()
            down_pcd, index_matrix, index = pcd.voxel_down_sample_and_trace(
                bbox_size.min() / self.downsample_resolution, pcd.get_min_bound(), pcd.get_max_bound(), False
            )

            # Downsample and average feature:
            dim = clt_descriptors.shape[0]
            clt_descriptors = clt_descriptors.view(dim, -1, self.num_leaf)
            clt_scores = clt_scores.view(-1, self.num_leaf, 1)

            avg_features3d_list = []
            avg_scores_list = []
            avg_clt_descriptors = []
            avg_clt_scores = []
            inverse_mapping = -1 * torch.ones((keypoints3d.shape[0],)) # All -1
            for i, ind in enumerate(index):
                ind = torch.from_numpy(np.array(ind)).to(torch.long)
                feature = torch.mean(avg_descriptors3d[:, ind], dim=1, keepdim=False) # d
                scores = torch.mean(avg_scores[ind, :], dim=0, keepdim=False) # d
                avg_features3d_list.append(feature)
                avg_scores_list.append(scores)

                # FIXME: feature2d_db is not a good implementation, only select index 0 for GATs!
                avg_clt_descriptors.append(clt_descriptors[:, ind[0], :]) # D*num_leaf
                avg_clt_scores.append(clt_scores[ind[0],:]) # 

                inverse_mapping[ind] = i

            avg_descriptors3d = torch.stack(avg_features3d_list, dim=-1) # D*N
            avg_scores = torch.stack(avg_scores_list, dim=0)
            avg_clt_descriptors = torch.stack(avg_clt_descriptors, dim=1) # D*N*num_leaf
            avg_clt_scores = torch.stack(avg_clt_scores, dim=0) # N*num_leaf*1

            clt_descriptors = avg_clt_descriptors.view(dim, -1)
            clt_scores = avg_clt_scores.view(-1, 1)
            keypoints3d = torch.from_numpy(np.array(down_pcd.points)).to(torch.float)

            if assignmatrix is not None:
                # Remapping assignmatrix
                assignmatrix = assignmatrix.long()
                assignmatrix[1,:] = inverse_mapping[assignmatrix[1,:]]
                unique, unique_index = np.unique(assignmatrix[1,:], return_index=True)
                # unique, unique_index = torch.unique(assignmatrix[1,:], return_inverse=True)
                assignmatrix = assignmatrix[:, unique_index]
            
        num_3d_orig = keypoints3d.shape[0]

        if pad:
            if self.split == "train":
                if assignmatrix is not None:
                    (
                        keypoints3d,
                        assignmatrix,
                        padding_index,
                    ) = data_utils.pad_keypoints3d_according_to_assignmatrix(
                        keypoints3d, self.shape3d, assignmatrix=assignmatrix
                    )
                    (
                        avg_descriptors3d,
                        avg_scores,
                    ) = data_utils.pad_features3d_according_to_assignmatrix(
                        avg_descriptors3d, avg_scores, self.shape3d, padding_index
                    )
                    (
                        clt_descriptors,
                        clt_scores,
                    ) = data_utils.pad_features3d_leaves_according_to_assignmatrix(
                        clt_descriptors,
                        clt_scores,
                        num_3d_orig,
                        self.shape3d,
                        num_leaf=self.num_leaf,
                        padding_index=padding_index,
                    )

                else:
                    keypoints3d = data_utils.pad_keypoints3d_top_n(
                        keypoints3d, self.shape3d
                    )
                    avg_descriptors3d, avg_scores = data_utils.pad_features3d_top_n(
                        avg_descriptors3d, avg_scores, self.shape3d
                    )
                    clt_descriptors, clt_scores = data_utils.pad_features3d_leaves_top_n(
                        clt_descriptors,
                        clt_scores,
                        idxs,
                        self.shape3d,
                        num_leaf=self.num_leaf,
                    )
            else:
                (keypoints3d, padding_index,) = data_utils.pad_keypoints3d_random(
                    keypoints3d, self.shape3d
                )
                (avg_descriptors3d, avg_scores,) = data_utils.pad_features3d_random(
                    avg_descriptors3d, avg_scores, self.shape3d, padding_index
                )
                (
                    clt_descriptors,
                    clt_scores,
                ) = data_utils.pad_features3d_leaves_random(
                    clt_descriptors,
                    clt_scores,
                    idxs,
                    self.shape3d,
                    num_leaf=self.num_leaf,
                    padding_index=padding_index,
                )

        return (
            keypoints3d,
            avg_descriptors3d,
            avg_scores,
            clt_descriptors,
            clt_scores,
            num_3d_orig,
            assignmatrix,  # Update assignmatrix
        )

    def build_assignmatrix(
        self, keypoints2D_coarse, keypoints2D_fine, assign_matrix, pad=True
    ):
        """
        Build assign matrix for coarse and fine
        Coarse assign matrix: store 0 or 1
        Fine matrix: store corresponding 2D fine location in query image of the matched coarse grid point (N*M*2)
        """

        """ Reshape assign matrix (from 2xk to nxm)"""
        assign_matrix = assign_matrix.long()

        if pad:
            conf_matrix = torch.zeros(
                self.shape3d, self.n_query_coarse_grid, dtype=torch.int16
            )  # [n_pointcloud, n_coarse_grid]

            fine_location_matrix = torch.full(
                (self.shape3d, self.n_query_coarse_grid, 2), -50, dtype=torch.float
            )

            # Padding
            valid = assign_matrix[1] < self.shape3d
            assign_matrix = assign_matrix[:, valid]

            # Get grid coordinate for query image
            keypoints_idx = assign_matrix[0]
            keypoints2D_coarse_selected = keypoints2D_coarse[keypoints_idx]

            keypoints2D_fine_selected = keypoints2D_fine[keypoints_idx]

            # Get j_id of coarse keypoints in grid
            keypoints2D_coarse_selected_rescaled = (
                keypoints2D_coarse_selected
                / self.query_img_scale[[1, 0]]
                * self.coarse_scale
            )
            keypoints2D_coarse_selected_rescaled = (
                keypoints2D_coarse_selected_rescaled.round()
            )
            unique, counts = np.unique(
                keypoints2D_coarse_selected_rescaled, return_counts=True, axis=0
            )
            if unique.shape[0] != keypoints2D_coarse_selected_rescaled.shape[0]:
                logger.warning("Keypoints duplicate! Problem exists")

            j_ids = (
                keypoints2D_coarse_selected_rescaled[:, 1] * self.w_c  # y
                + keypoints2D_coarse_selected_rescaled[:, 0]  # x
            )
            j_ids = j_ids.long()

            invalid_mask = j_ids > conf_matrix.shape[1]
            j_ids = j_ids[~invalid_mask]
            assign_matrix = assign_matrix[:, ~invalid_mask]
            keypoints2D_fine_selected = keypoints2D_fine_selected[~invalid_mask]
            # if invalid_mask.sum() != 0:
            #     logger.warning(f"{invalid_mask.sum()} points locate outside frame")
            # x, y = j_ids % self.w_c, j_ids // self.w_c

            conf_matrix[assign_matrix[1], j_ids] = 1
            fine_location_matrix[assign_matrix[1], j_ids] = keypoints2D_fine_selected

        else:
            raise NotImplementedError

        return conf_matrix, fine_location_matrix

    def get_intrin_by_color_pth(self, img_path):
        intrin_path = img_path.replace("/color/", "/intrin_ba/").replace(".png", ".txt")
        K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]
        return K_crop

    def get_gt_pose_by_color_pth(self, img_path):
        gt_pose_path = img_path.replace("/color/", "/poses_ba/").replace(".png", ".txt")
        pose_gt = torch.from_numpy(np.loadtxt(gt_pose_path))  # [4*4]
        return pose_gt

    def read_anno(self, img_id):
        """
        read image, 2d info and 3d info.
        pad 2d info and 3d info to a constant size.
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        color_path = self.coco.loadImgs(int(img_id))[0]["img_file"]

        if (
            self.path_prefix_substitute_2D_source[0] is not None
            and self.path_prefix_substitute_2D_aim[0] is not None
        ):
            if self.path_prefix_substitute_2D_source[0] in color_path:
                color_path = color_path.replace(
                    self.path_prefix_substitute_2D_source[0],
                    self.path_prefix_substitute_2D_aim[0],
                )

        query_img, query_img_scale, query_img_mask = read_grayscale(
            color_path,
            resize=self.img_resize,
            pad_to=self.img_resize if self.img_pad else None,
            ret_scales=True,
            ret_pad_mask=True,
            df=self.df,
        )

        self.h_origin = query_img.shape[1] * query_img_scale[0]
        self.w_origin = query_img.shape[2] * query_img_scale[1]
        self.query_img_scale = query_img_scale
        self.h_i = query_img.shape[1]
        self.w_i = query_img.shape[2]
        self.h_c = int(self.h_i * self.coarse_scale)
        self.w_c = int(self.w_i * self.coarse_scale)

        self.n_query_coarse_grid = int(self.h_c * self.w_c)

        data = {}
        if query_img_mask is not None:
            data.update({"query_image_mask": query_img_mask})  # [h*w]

        if self.load_pose_gt:
            K_crop = self.get_intrin_by_color_pth(color_path)
            pose_gt = self.get_gt_pose_by_color_pth(color_path)

            data.update({"query_intrinsic": K_crop, "query_pose_gt": pose_gt})

        if self.split == "train":
            # For query image GT correspondences
            anno2d_file = anno["anno2d_file"]

            if (
                self.path_prefix_substitute_2D_source[0] is not None
                and self.path_prefix_substitute_2D_aim[0] is not None
            ):
                if self.path_prefix_substitute_2D_source[0] in anno2d_file:
                    anno2d_file = anno2d_file.replace(
                        self.path_prefix_substitute_2D_source[0],
                        self.path_prefix_substitute_2D_aim[0],
                    )

            anno2d_coarse_file = anno2d_file.replace(
                "/anno_loftr/", "/anno_loftr_coarse/"
            )
            (
                keypoints2d_coarse,
                scores2d,
                assign_matrix,
                num_2d_orig,
            ) = self.read_anno2d(anno2d_coarse_file)

            # NOTE: use 3d point cloud project to 2D to make fine match instead of feature track later
            # (
            #     keypoints2d_fine,
            #     scores2d,
            #     assign_matrix_fine,
            #     num_2d_orig,
            # ) = self.read_anno2d(anno2d_file)

        else:
            assign_matrix = None

        idxs_file = anno["idxs_file"]
        avg_anno3d_file = anno["avg_anno3d_file"]
        collect_anno3d_file = anno["collect_anno3d_file"]

        if (
            self.path_prefix_substitute_3D_source[0] is not None
            and self.path_prefix_substitute_3D_aim[0] is not None
        ):
            if self.path_prefix_substitute_3D_source[0] in idxs_file:
                idxs_file = idxs_file.replace(
                    self.path_prefix_substitute_3D_source[0],
                    self.path_prefix_substitute_3D_aim[0],
                )
                avg_anno3d_file = avg_anno3d_file.replace(
                    self.path_prefix_substitute_3D_source[0],
                    self.path_prefix_substitute_3D_aim[0],
                )
                collect_anno3d_file = collect_anno3d_file.replace(
                    self.path_prefix_substitute_3D_source[0],
                    self.path_prefix_substitute_3D_aim[0],
                )

        (
            keypoints3d,
            avg_descriptors3d,
            avg_scores3d,
            clt_descriptors2d,
            clt_scores2d,
            num_3d_orig,
            assign_matrix,
        ) = self.read_anno3d(
            avg_anno3d_file,
            collect_anno3d_file,
            idxs_file,
            pad=self.pad,
            assignmatrix=assign_matrix,
        )

        data.update(
            {
                "keypoints3d": keypoints3d,  # [n2, 3]
                "descriptors3d_db": avg_descriptors3d,  # [dim, n2]
                "descriptors2d_db": clt_descriptors2d,  # [dim, n2 * num_leaf]
                "scores3d_db": avg_scores3d.squeeze(1),  # [n2]
                "scores2d_db": clt_scores2d.squeeze(1),  # [n2 * num_leaf]
                "query_image": query_img,  # [1*h*w]
                "query_image_scale": query_img_scale,  # [2]
            }
        )

        if self.split == "train":
            assign_matrix = assign_matrix.long()
            mkpts_3d = keypoints3d[assign_matrix[1, :]]  # N*3
            R = pose_gt[:3, :3].to(torch.float)  # 3*3
            t = pose_gt[:3, [3]].to(torch.float)  # 3*1
            K_crop = K_crop.to(torch.float)
            keypoints2d_fine = torch.zeros(
                (keypoints2d_coarse.shape[0], 2), dtype=torch.float
            )

            # Project 3D pointcloud to make fine GT
            mkpts_3d_camera = R @ mkpts_3d.transpose(1, 0) + t
            mkpts_proj = (K_crop @ mkpts_3d_camera).transpose(1, 0)  # N*3
            mkpts_proj = mkpts_proj[:, :2] / (mkpts_proj[:, [2]] + 1e-6)

            # if self.downsample:
            mkpts_proj_rounded = (mkpts_proj / int(1/self.coarse_scale)).round() * int(1/self.coarse_scale)
            invalid = (mkpts_proj_rounded[:,0] < 0) | (mkpts_proj_rounded[:, 0] > query_img.shape[-1] - 1) | (mkpts_proj_rounded[:, 1] < 0) | (mkpts_proj_rounded[:, 1] > query_img.shape[-2] - 1)
            mkpts_proj_rounded = mkpts_proj_rounded[~invalid]
            mkpts_proj = mkpts_proj[~invalid]
            assign_matrix = assign_matrix[:, ~invalid]

            keypoint2d_coarse_original = keypoints2d_coarse[assign_matrix[0,:]]
            unique, index = np.unique(mkpts_proj_rounded, return_index=True, axis=0)

            mkpts_proj_rounded = mkpts_proj_rounded[index]
            mkpts_proj = mkpts_proj[index]
            assign_matrix = assign_matrix[:, index]
            keypoints2d_coarse[assign_matrix[0,:]] = mkpts_proj_rounded


            keypoints2d_fine[assign_matrix[0, :]] = mkpts_proj


            (conf_matrix, fine_location_matrix) = self.build_assignmatrix(
                keypoints2d_coarse, keypoints2d_fine, assign_matrix, pad=self.pad
            )

            data.update(
                {
                    # GT
                    "conf_matrix_gt": conf_matrix,  # [n_point_cloud, n_query_coarse_grid] Used for coarse GT
                    "fine_location_matrix_gt": fine_location_matrix,  # [n_point_cloud, n_query_coarse_grid, 2] (x,y)
                }
            )

            # # NOTE: For fine match debug TODO: remove
            # gt_mkpts_3d = -1 * torch.ones((keypoints3d.shape[0],), dtype=torch.long) # Index to keypoints2d_fine
            # gt_mkpts_3d[assign_matrix[1, :]] = assign_matrix[0, :]
            # data.update(
            #     {"gt_mkpts_3d_idx": gt_mkpts_3d, 'keypoints2d_fine_gt': keypoints2d_fine}
            # )

        # mkpt fine check:
        # # TODO: remove
        # if self.split == 'train':
        #     mkpts_2d_fine = keypoints2d_fine[assign_matrix[0,:]] # N*2
        #     mkpts_3d = keypoints3d[assign_matrix[1,:]] # N*3

        #     mkpts_3d_camera = R @ mkpts_3d.transpose(1,0) + t
        #     mkpts_proj = (K_crop @ mkpts_3d_camera).transpose(1,0) # N*3
        #     mkpts_proj = mkpts_proj[:, :2] / (mkpts_proj[:, [2]] + 1e-4)
        #     coarse_fine_offset = mkpts_2d_fine - keypoints2d_coarse[assign_matrix[0, :]]
        #     diff = mkpts_proj - mkpts_2d_fine
        #     diff =diff
        return data

    def __getitem__(self, index):
        img_id = self.anns[index]

        data = self.read_anno(img_id)
        return data

    def __len__(self):
        return len(self.anns)
