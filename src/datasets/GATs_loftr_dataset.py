import cv2

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
        img_resize=640,
        coarse_scale=1 / 8,
        df=8,
        shape2d=2000,
        shape3d=10000,
    ):
        super(Dataset, self).__init__()

        self.coco = COCO(anno_file)
        self.anns = np.array(self.coco.getImgIds())
        self.num_leaf = num_leaf

        # 3D point cloud part
        self.pad = pad
        self.shape2d = shape2d
        self.shape3d = shape3d

        # 2D query image part
        self.img_pad = img_pad
        self.img_resize = img_resize
        self.df = df
        self.coarse_scale = coarse_scale

    def read_anno2d(self, anno2d_file, height, width, pad=True):
        """ Read (and pad) 2d info"""
        with open(anno2d_file, "r") as f:
            data = json.load(f)

        keypoints2d = torch.Tensor(data["keypoints2d"])  # [n, 2]
        descriptors2d = torch.Tensor(data["descriptors2d"])  # [dim, n]
        scores2d = torch.Tensor(data["scores2d"])  # [n, 1]
        assign_matrix = torch.Tensor(data["assign_matrix"])  # [2, k]

        num_2d_orig = keypoints2d.shape[0]

        if pad:
            keypoints2d, descriptors2d, scores2d = data_utils.pad_keypoints2d_random(
                keypoints2d, descriptors2d, scores2d, height, width, self.shape2d
            )
        return keypoints2d, descriptors2d, scores2d, assign_matrix, num_2d_orig

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

        height = query_img.shape[1] * query_img_scale[0]
        width = query_img.shape[2] * query_img_scale[1]

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
        (
            keypoints2d,
            descriptors2d,
            scores2d,
            assign_matrix,
            num_2d_orig,
        ) = self.read_anno2d(anno2d_file, height, width, pad=self.pad)

        conf_matrix = data_utils.reshape_assign_matrix(
            assign_matrix,
            num_2d_orig,
            num_3d_orig,
            self.shape2d,
            self.shape3d,
            pad=True,
        )
        conf_matrix = conf_matrix

        data = {
            "keypoints3d": keypoints3d,  # [n2, 3]
            "descriptors3d_db": avg_descriptors3d,  # [dim, n2]
            "descriptors2d_db": clt_descriptors2d,  # [dim, n2 * num_leaf]
            "scores3d_db": avg_scores3d, # [n2, 1]
            "scores2d_db": clt_descriptors2d, # [n2 * num_leaf, 1]

            # TODO: remove
            # "keypoints2d_query": keypoints2d,  # [n1, 2] query image
            # "descriptors2d_query": descriptors2d,  # [dim, n1]
            # "image_size": torch.Tensor([height, width]),

            "query_image": query_img, # [1*h*w]
            "query_image_scale": query_img_scale, # [2]

            # GT
            "conf_matrix": conf_matrix
        }
        if query_img_mask is not None:
            data.update({"query_image_mask": query_img_mask}) # [h*w]
        return data

    def __getitem__(self, index):
        img_id = self.anns[index]

        data = self.read_anno(img_id)
        return data

    def __len__(self):
        return len(self.anns)
