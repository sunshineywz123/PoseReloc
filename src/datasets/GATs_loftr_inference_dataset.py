from loguru import logger
import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset

from .utils import read_grayscale
from src.utils import data_utils


class GATs_loftr_inference_dataset(Dataset):
    def __init__(
        self,
        sfm_dir,
        image_paths,
        shape3d,
        num_leaf,
        img_pad=False,
        img_resize=512,
        df=8,
        pad=True,
        load_pose_gt=True,
        n_images=None # Used for debug
    ) -> None:
        super().__init__()

        self.shape3d = shape3d
        self.num_leaf = num_leaf
        self.pad = pad
        self.image_paths = image_paths[::int(len(image_paths) / n_images)] if n_images is not None else image_paths
        self.img_pad =img_pad
        self.img_resize= img_resize
        self.df = df
        self.load_pose_gt = load_pose_gt

        # Load pointcloud and point feature
        avg_anno_3d_path, clt_anno_3d_path, idxs_path = self.get_default_paths(sfm_dir)
        (
            self.keypoints3d,
            self.avg_descriptors3d,
            self.avg_scores3d,
            self.clt_descriptors2d,
            self.clt_scores2d,
            self.num_3d_orig,
        ) = self.read_anno3d(
            avg_anno_3d_path,
            clt_anno_3d_path,
            idxs_path,
            pad=self.pad,
        )
    

    def get_default_paths(self, sfm_model_dir):
        anno_dir = osp.join(sfm_model_dir, f"anno")
        avg_anno_3d_path = osp.join(anno_dir, "anno_3d_average.npz")
        clt_anno_3d_path = osp.join(anno_dir, "anno_3d_collect.npz")
        idxs_path = osp.join(anno_dir, "idxs.npy")

        return avg_anno_3d_path, clt_anno_3d_path, idxs_path

    def get_intrin_by_color_pth(self, img_path):
        intrin_path = img_path.replace("/color/", "/intrin_ba/").replace(".png", ".txt")
        K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]
        return K_crop

    def get_gt_pose_by_color_pth(self, img_path):
        gt_pose_path = img_path.replace("/color/", "/poses_ba/").replace(".png", ".txt")
        pose_gt = torch.from_numpy(np.loadtxt(gt_pose_path))  # [4*4]
        return pose_gt

    def read_anno3d(
        self, avg_anno3d_file, clt_anno3d_file, idxs_file, pad=True
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

        num_3d_orig = keypoints3d.shape[0]

        clt_descriptors, clt_scores = data_utils.build_features3d_leaves(
            clt_descriptors, clt_scores, idxs, num_leaf=self.num_leaf
        )
        if pad:
            (
                keypoints3d,
                padding_index,
            ) = data_utils.pad_keypoints3d_random(
                keypoints3d, self.shape3d
            )
            (
                avg_descriptors3d,
                avg_scores,
            ) = data_utils.pad_features3d_random(
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
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        query_img, query_img_scale, query_img_mask = read_grayscale(
            image_path,
            resize=self.img_resize,
            pad_to=self.img_resize if self.img_pad else None,
            ret_scales=True,
            ret_pad_mask=True,
            df=self.df,
        )

        data = {}

        data.update(
            {
                "keypoints3d": self.keypoints3d[None],  # [n2, 3]
                "descriptors3d_db": self.avg_descriptors3d[None],  # [dim, n2]
                "descriptors2d_db": self.clt_descriptors2d[None],  # [dim, n2 * num_leaf]
                "scores3d_db": self.avg_scores3d.squeeze(1)[None],  # [n2]
                "scores2d_db": self.clt_descriptors2d.squeeze(1)[None],  # [n2 * num_leaf]
                "query_image": query_img[None],  # [1*h*w]
                "query_image_scale": query_img_scale[None],  # [2]
                'query_image_path': image_path
            }
        )

        if query_img_mask is not None:
            data.update({"query_image_mask": query_img_mask[None]})  # [h*w]

        if self.load_pose_gt:
            K_crop = self.get_intrin_by_color_pth(image_path)
            pose_gt = self.get_gt_pose_by_color_pth(image_path)
            data.update({"query_intrinsic": K_crop[None], "query_pose_gt": pose_gt[None]})
        
        return data