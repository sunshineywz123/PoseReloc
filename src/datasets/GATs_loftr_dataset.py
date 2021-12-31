import math
import cv2
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
        load_pose_gt=False,
    ):
        super(Dataset, self).__init__()

        self.coco = COCO(anno_file)
        self.anns = np.array(self.coco.getImgIds())
        logger.info(f"Use {percent * 100}% data")
        self.anns = self.anns[: math.floor(len(self.anns) * percent)]
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

    def read_anno2d(self, anno2d_file):
        """ Read (and pad) 2d info"""
        with open(anno2d_file, "r") as f:
            data = json.load(f)

        keypoints2d = torch.Tensor(data["keypoints2d"])  # [n, 2]
        scores2d = torch.Tensor(data["scores2d"])  # [n, 1]
        assign_matrix = torch.Tensor(data["assign_matrix"])  # [2, k]

        num_2d_orig = keypoints2d.shape[0]

        # if pad:
        #     keypoints2d, descriptors2d, scores2d = data_utils.pad_keypoints2d_random(
        #         keypoints2d, descriptors2d, scores2d, height, width, self.shape2d
        #     )
        return keypoints2d, scores2d, assign_matrix, num_2d_orig

    def read_anno3d(self, avg_anno3d_file, clt_anno3d_file, idxs_file, pad=True):
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

        num_3d_orig = keypoints3d.shape[0]
        if pad:
            keypoints3d = data_utils.pad_keypoints3d_random(keypoints3d, self.shape3d)
            avg_descriptors3d, avg_scores = data_utils.pad_features3d_random(
                avg_descriptors3d, avg_scores, self.shape3d
            )
            clt_descriptors, clt_scores = data_utils.build_features3d_leaves(
                clt_descriptors, clt_scores, idxs, self.shape3d, num_leaf=self.num_leaf
            )
        return (
            keypoints3d,
            avg_descriptors3d,
            avg_scores,
            clt_descriptors,
            clt_scores,
            num_3d_orig,
        )

    def reshape_assignmatrix(
        self, keypoints2D_coarse, keypoints2D_fine, assign_matrix, pad=True
    ):
        """ Reshape assign matrix (from 2xk to nxm)"""
        assign_matrix = assign_matrix.long()

        if pad:
            conf_matrix = torch.zeros(
                self.shape3d, self.n_query_coarse_grid, dtype=torch.int16
            )

            # Padding
            valid = assign_matrix[1] < self.shape3d
            assign_matrix = assign_matrix[:, valid]  # [n_pointcloud, n_coarse_grid]

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
                keypoints2D_coarse_selected_rescaled[:, 0] * self.w_c
                + keypoints2D_coarse_selected_rescaled[:, 1]
            )
            j_ids = j_ids.long()

            conf_matrix[assign_matrix[1], j_ids] = 1
            # conf_matrix[orig_shape2d:] = -1
            # conf_matrix[:, orig_shape3d:] = -1
        else:
            raise NotImplementedError

        return j_ids, keypoints2D_fine_selected, assign_matrix[1], conf_matrix

    def get_intrin_by_color_pth(self, img_path):
        intrin_path = img_path.replace("/color_crop/", "/intrin_crop_ba/").replace(
            ".png", ".txt"
        )
        K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]
        return K_crop

    def get_gt_pose_by_color_pth(self, img_path):
        gt_pose_path = img_path.replace("/color_crop/", "/poses_ba/").replace(
            ".png", ".txt"
        )
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

        idxs_file = anno["idxs_file"]
        avg_anno3d_file = anno["avg_anno3d_file"]
        collect_anno3d_file = anno["collect_anno3d_file"]
        (
            keypoints3d,
            avg_descriptors3d,
            avg_scores3d,
            clt_descriptors2d,
            clt_scores2d,
            num_3d_orig,
        ) = self.read_anno3d(
            avg_anno3d_file, collect_anno3d_file, idxs_file, pad=self.pad
        )
        anno2d_file = anno["anno2d_file"]

        anno2d_coarse_file = anno2d_file.replace("/anno_loftr", "/anno_loftr_coarse")
        # FIXME: not an efficient solution: Load feature twice however no use! change sfm save keypoints' feature part
        (
            keypoints2d_coarse,
            scores2d,
            assign_matrix,
            num_2d_orig,
        ) = self.read_anno2d(anno2d_coarse_file)

        (
            keypoints2d_fine,
            scores2d,
            assign_matrix,
            num_2d_orig,
        ) = self.read_anno2d(anno2d_file)

        (
            j_ids,
            keypoints2D_fine_selected,
            points3D_idx_padded,
            conf_matrix,
        ) = self.reshape_assignmatrix(keypoints2d_coarse, keypoints2d_fine, assign_matrix, pad=self.pad)

        data = {
            "keypoints3d": keypoints3d,  # [n2, 3]
            "descriptors3d_db": avg_descriptors3d,  # [dim, n2]
            "descriptors2d_db": clt_descriptors2d,  # [dim, n2 * num_leaf]
            "scores3d_db": avg_scores3d.squeeze(1),  # [n2]
            "scores2d_db": clt_descriptors2d.squeeze(1),  # [n2 * num_leaf]
            "query_image": query_img,  # [1*h*w]
            "query_image_scale": query_img_scale,  # [2]
            # GT
            "conf_matrix_gt": conf_matrix,  # [n_point_cloud, n_query_coarse_grid] Used for coarse GT
            "mkpts3D_idx_gt": points3D_idx_padded, # [N, 3]
            "mkpts2D_gt": keypoints2D_fine_selected # [N,2] # Used for fine GT
        }
        if query_img_mask is not None:
            data.update({"query_image_mask": query_img_mask})  # [h*w]

        if self.load_pose_gt:
            K_crop = self.get_intrin_by_color_pth(color_path)
            pose_gt = self.get_gt_pose_by_color_pth(color_path)

            data.update({"query_intrinsic": K_crop, "query_pose_gt": pose_gt})
        return data

    def __getitem__(self, index):
        img_id = self.anns[index]

        data = self.read_anno(img_id)
        return data

    def __len__(self):
        return len(self.anns)
