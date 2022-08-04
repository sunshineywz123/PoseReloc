from loguru import logger
import os
import os.path as osp
import torch
import numpy as np
import cv2
import torch.nn.functional as F
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
        load_3d_coarse=True,
        img_pad=False,
        img_resize=512,
        df=8,
        pad=True,
        load_pose_gt=True,
        coarse_scale=1/8,
        n_images=None,  # Used for debug
    ) -> None:
        super().__init__()

        self.shape3d = shape3d
        self.num_leaf = num_leaf
        self.pad = pad
        self.image_paths = (
            image_paths[:: int(len(image_paths) / n_images)]
            if n_images is not None
            else image_paths
        )
        logger.info(f'Will process:{len(self.image_paths)} images ')
        self.img_pad = img_pad
        self.img_resize = img_resize
        self.df = df
        self.load_pose_gt = load_pose_gt
        self.load_img_mask=False
        self.coarse_scale = coarse_scale

        # Load pointcloud and point feature
        avg_anno_3d_path, clt_anno_3d_path, idxs_path = self.get_default_paths(sfm_dir)
        (
            self.keypoints3d,
            self.avg_descriptors3d,
            self.avg_coarse_descriptors3d,
            self.avg_scores3d,
            self.clt_descriptors2d,
            self.clt_coarse_descriptors2d,
            self.clt_scores2d,
            self.num_3d_orig,
        ) = self.read_anno3d(
            avg_anno_3d_path, clt_anno_3d_path, idxs_path, pad=self.pad, load_3d_coarse=load_3d_coarse
        )

    def get_default_paths(self, sfm_model_dir):
        anno_dir = osp.join(sfm_model_dir, f"anno")
        avg_anno_3d_path = osp.join(anno_dir, "anno_3d_average.npz")
        clt_anno_3d_path = osp.join(anno_dir, "anno_3d_collect.npz")
        idxs_path = osp.join(anno_dir, "idxs.npy")

        return avg_anno_3d_path, clt_anno_3d_path, idxs_path

    def get_intrin_by_color_pth(self, img_path):
        image_dir_name = osp.basename(osp.dirname(img_path))
        object_2D_detector = "GT"
        if "_" in image_dir_name and "_full" not in image_dir_name:
            object_2D_detector = image_dir_name.split("_", 1)[1]
        
        if object_2D_detector == "GT":
            intrin_name = 'intrin_ba'
        elif object_2D_detector == 'SPP+SPG':
            intrin_name = 'intrin_SPP+SPG'
        elif object_2D_detector == "loftr":
            intrin_name = 'intrin_loftr'
        else:
            raise NotImplementedError

        img_ext = osp.splitext(img_path)[1]
        intrin_path = img_path.replace("/"+image_dir_name+"/", "/"+intrin_name+"/").replace(img_ext, ".txt")
        assert osp.exists(intrin_path), f"{intrin_path}"
        K_crop = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]
        return K_crop

    def get_intrin_original_by_color_pth(self, img_path):
        image_dir_name = osp.basename(osp.dirname(img_path))

        img_ext = osp.splitext(img_path)[1]
        intrin_path = img_path.replace("/"+image_dir_name+"/", "/intrin/").replace(img_ext, ".txt")
        assert osp.exists(intrin_path), f"{intrin_path}"
        K = torch.from_numpy(np.loadtxt(intrin_path))  # [3*3]
        return K

    def get_gt_pose_by_color_pth(self, img_path):
        image_dir_name = osp.basename(osp.dirname(img_path))
        img_ext = osp.splitext(img_path)[1]
        gt_pose_path = img_path.replace("/" + image_dir_name + "/", "/poses_ba/").replace(img_ext, ".txt")
        assert osp.exists(gt_pose_path), f"{gt_pose_path}"
        pose_gt = torch.from_numpy(np.loadtxt(gt_pose_path))  # [4*4]
        return pose_gt

    def read_anno3d(
        self, avg_anno3d_file, clt_anno3d_file, idxs_file, pad=True, load_3d_coarse=True
    ):
        """ Read(and pad) 3d info"""
        avg_data = np.load(avg_anno3d_file)
        clt_data = np.load(clt_anno3d_file)
        idxs = np.load(idxs_file)

        keypoints3d = torch.Tensor(clt_data["keypoints3d"])  # [m, 3]
        avg_descriptors3d = torch.Tensor(avg_data["descriptors3d"])  # [dim, m]
        clt_descriptors = torch.Tensor(clt_data["descriptors3d"])  # [dim, k]
        avg_scores = torch.Tensor(avg_data["scores3d"])  # [m, 1]
        clt_scores = torch.Tensor(clt_data["scores3d"])  # [k, 1]

        clt_descriptors, clt_scores = data_utils.build_features3d_leaves(
            clt_descriptors, clt_scores, idxs, num_leaf=self.num_leaf
        )

        num_3d_orig = keypoints3d.shape[0]

        if load_3d_coarse:
            avg_anno3d_coarse_file = (
                osp.splitext(avg_anno3d_file)[0]
                + "_coarse"
                + osp.splitext(avg_anno3d_file)[1]
            )
            clt_anno3d_coarse_file = (
                osp.splitext(clt_anno3d_file)[0]
                + "_coarse"
                + osp.splitext(clt_anno3d_file)[1]
            )
            avg_coarse_data = np.load(avg_anno3d_coarse_file)
            clt_coarse_data = np.load(clt_anno3d_coarse_file)
            avg_coarse_descriptors3d = torch.Tensor(
                avg_coarse_data["descriptors3d"]
            )  # [dim, m]
            clt_coarse_descriptors = torch.Tensor(
                clt_coarse_data["descriptors3d"]
            )  # [dim, k]
            avg_coarse_scores = torch.Tensor(avg_coarse_data["scores3d"])  # [m, 1]
            clt_coarse_scores = torch.Tensor(clt_coarse_data["scores3d"])  # [k, 1]

            (
                clt_coarse_descriptors,
                clt_coarse_scores,
            ) = data_utils.build_features3d_leaves(
                clt_coarse_descriptors, clt_coarse_scores, idxs, num_leaf=self.num_leaf
            )
        else:
            avg_coarse_descriptors3d = None
            clt_coarse_descriptors = None

        if pad:
            (keypoints3d, padding_index,) = data_utils.pad_keypoints3d_random(
                keypoints3d, self.shape3d
            )
            (avg_descriptors3d, avg_scores,) = data_utils.pad_features3d_random(
                avg_descriptors3d, avg_scores, self.shape3d, padding_index
            )
            (clt_descriptors, clt_scores,) = data_utils.pad_features3d_leaves_random(
                clt_descriptors,
                clt_scores,
                idxs,
                self.shape3d,
                num_leaf=self.num_leaf,
                padding_index=padding_index,
            )

            if avg_coarse_descriptors3d is not None:
                (
                    avg_coarse_descriptors3d,
                    avg_coarse_scores,
                ) = data_utils.pad_features3d_random(
                    avg_coarse_descriptors3d,
                    avg_coarse_scores,
                    self.shape3d,
                    padding_index,
                )
                (
                    clt_coarse_descriptors,
                    clt_coarse_scores,
                ) = data_utils.pad_features3d_leaves_random(
                    clt_coarse_descriptors,
                    clt_coarse_scores,
                    idxs,
                    self.shape3d,
                    num_leaf=self.num_leaf,
                    padding_index=padding_index,
                )

        return (
            keypoints3d,
            avg_descriptors3d,
            avg_coarse_descriptors3d,
            avg_scores,
            clt_descriptors,
            clt_coarse_descriptors,
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
        self.h_origin = query_img.shape[1] * query_img_scale[0]
        self.w_origin = query_img.shape[2] * query_img_scale[1]
        self.query_img_scale = query_img_scale
        self.h_i = query_img.shape[1]
        self.w_i = query_img.shape[2]
        self.h_c = int(self.h_i * self.coarse_scale)
        self.w_c = int(self.w_i * self.coarse_scale)
        
        image_dir_name = osp.basename(osp.dirname(image_path))
        if self.load_img_mask:
            img_ext = osp.splitext(image_path)[1]
            reproj_box3d = np.loadtxt(
                image_path.replace('/'+image_dir_name+'/', "/reproj_box/").replace(
                    img_ext, ".txt"
                )
            ).astype(int)
            x0, y0 = reproj_box3d.min(0)
            x1, y1 = reproj_box3d.max(0)

            original_img = cv2.imread(image_path.replace('/'+image_dir_name+'/', "/color_full/"))
            assert (
                original_img is not None
            ), f"color full path: {image_path.replace('/'+image_dir_name+'/', '/color_full/')} not exists"
            origin_h, origin_w = original_img.shape[:2]  # H, W before crop
            original_img_fake = np.ones((origin_h, origin_w))  # All white
            box = np.array([x0, y0, x1, y1])
            resize_shape = np.array([y1 - y0, x1 - x0])
            image_crop, _ = data_utils.get_image_crop_resize(
                original_img_fake, box, resize_shape
            )

            box_new = np.array([0, 0, x1 - x0, y1 - y0])
            resize_shape = np.array([self.h_origin, self.w_origin])
            image_crop, _ = data_utils.get_image_crop_resize(
                image_crop, box_new, resize_shape
            )
            crop_border_mask = image_crop != 0
            crop_border_mask_rescaled = (
                F.interpolate(
                    torch.from_numpy(crop_border_mask)[None][None].to(torch.float),
                    scale_factor=self.coarse_scale,
                )
                .squeeze()
                .to(torch.bool)
            )  # H_coarse * W_coarse
            query_img_mask = (
                crop_border_mask_rescaled
                if query_img_mask is None
                else query_img_mask & crop_border_mask_rescaled
            )

        data = {}

        desc2d_db_padding_mask = torch.sum(self.clt_descriptors2d==1, dim=0) != self.clt_descriptors2d.shape[0]
        data.update(
            {
                "keypoints3d": self.keypoints3d[None],  # [1, n2, 3]
                "descriptors3d_db": self.avg_descriptors3d[None],  # [1, dim, n2]
                "descriptors2d_db": self.clt_descriptors2d[
                    None
                ],  # [1, dim, n2 * num_leaf]
                "desc2d_db_padding_mask": desc2d_db_padding_mask, # [n2*num_leaf]
                "scores3d_db": self.avg_scores3d.squeeze(1)[None],  # [1, n2]
                "scores2d_db": self.clt_descriptors2d.squeeze(1)[
                    None
                ],  # [1, n2 * num_leaf]
                "query_image": query_img[None],  # [1*h*w]
                "query_image_scale": query_img_scale[None],  # [2]
                "query_image_path": image_path,
            }
        )

        if self.avg_coarse_descriptors3d is not None:
            desc2d_coarse_db_padding_mask = torch.sum(self.clt_coarse_descriptors2d==1, dim=0) != self.clt_coarse_descriptors2d.shape[0]
            data.update({
                "descriptors3d_coarse_db": self.avg_coarse_descriptors3d[None],  # [1, dim, n2]
                "descriptors2d_coarse_db": self.clt_coarse_descriptors2d[None],  # [1, dim, n2 * num_leaf]
                "desc2d_coarse_db_padding_mask": desc2d_coarse_db_padding_mask, # [n2*num_leaf]
            })

        if query_img_mask is not None:
            data.update({"query_image_mask": query_img_mask[None]})  # [h*w]

        if self.load_pose_gt:
            K_crop = self.get_intrin_by_color_pth(image_path)
            # FIXME: K is not real original image intrins, it is K_crop before BA.
            # However, for LINEMOD data, we store its original intrin in the intrin directory.
            K = self.get_intrin_original_by_color_pth(image_path)
            pose_gt = self.get_gt_pose_by_color_pth(image_path)
            data.update(
                {"query_intrinsic": K_crop[None], "query_pose_gt": pose_gt[None], "query_intrinsic_origin": K[None]}
            )

        return data
