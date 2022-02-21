from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from loguru import logger


class MatchingPairData(Dataset):
    """
    Construct image pair for LoFTR fine matching
    """

    def __init__(self, colmap_image_dataset) -> None:
        super().__init__()

        # Colmap info
        self.colmap_image_dataset = colmap_image_dataset
        self.colmap_frame_dict = colmap_image_dataset.colmap_frame_dict
        self.colmap_3ds = colmap_image_dataset.colmap_3ds
        self.colmap_images = colmap_image_dataset.colmap_images
        self.colmap_cameras = colmap_image_dataset.colmap_cameras

        self.warp_image1_by_homo = (
            True  # Wrap image1 to get more genernal local feature
        )

        self.all_pairs = []
        for colmap_frameID, colmap_frame_info in self.colmap_frame_dict.items():
            if colmap_frame_info["is_keyframe"]:
                for related_frameID in colmap_frame_info["related_frameID"]:
                    self.all_pairs.append([colmap_frameID, related_frameID])

    def buildDataPair(self, data0, data1):
        # data0: dict, data1: dict
        data = {}
        for i, data_part in enumerate([data0, data1]):
            for key, value in data_part.items():
                data[key + str(i)] = value
        assert (
            len(data) % 2 == 0
        ), "build data pair error! please check built data pair!"
        data["pair_names"] = (data["img_path0"], data["img_path1"])
        return data

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        left_img_id, right_img_id = self.all_pairs[index]  # colmap id

        # Get coarse matches
        left_kpts = []  # x,y
        right_kpts = []
        left_kpts_idx = []  # correspond to original index of left frame keypoints

        left_frame_info = self.colmap_frame_dict[left_img_id]
        valid_kpts_mask = left_frame_info["all_kpt_status"] >= 0
        valid_kpts_idxs = np.arange(left_frame_info["keypoints"].shape[0])[
            valid_kpts_mask
        ]
        valid_kpts = left_frame_info["keypoints"][valid_kpts_mask]
        related_3d_ids = left_frame_info["all_kpt_status"][valid_kpts_mask]
        for i, related_3d_id in enumerate(related_3d_ids.tolist()):
            related_index = np.argwhere(
                self.colmap_3ds[related_3d_id].image_ids == right_img_id
            )  # (1,1) or (0,1)
            if len(related_index) != 0:
                # successfully find!
                if len(related_index) != 1:
                    print(self.colmap_3ds[related_3d_id].image_ids)
                    # FIXME: Duplicate keypoints in right frame! Currenct solution is use first one!
                    related_index = related_index[0]
                    # import ipdb; ipdb.set_trace()
                point2d_idx = self.colmap_3ds[related_3d_id].point2D_idxs[
                    np.squeeze(related_index).tolist()
                ]  # int
                left_kpts.append(
                    valid_kpts[i]
                )  # FIXME: duplicate points in a feature track is counted twice!
                right_kpts.append(self.colmap_images[right_img_id].xys[point2d_idx])
                # print(f"right img id:{right_img_id}, point2d_idx:{point2d_idx}, point location: {self.colmap_images[right_img_id].xys[point2d_idx]}")

                # Record left keypoints index in original frame keypoints
                (
                    self_img_id,
                    self_kpt_idx,
                ) = self.colmap_image_dataset.point_cloud_assigned_imgID_kptID[
                    related_3d_id
                ]
                assert self_img_id == left_img_id
                if self_kpt_idx != valid_kpts_idxs[i]:
                    logger.warning("Duplicate point exists in keyframe")
                left_kpts_idx.append(valid_kpts_idxs[[i]])

        left_kpts = np.stack(left_kpts, axis=0)  # N*2
        right_kpts = np.stack(right_kpts, axis=0)  # N*2
        left_kpts_idx = np.concatenate(left_kpts_idx)  # N*1

        # Get images information
        left_id = self.colmap_image_dataset.colmapID2frameID_dict[
            left_img_id
        ]  # dataset image id
        right_id = self.colmap_image_dataset.colmapID2frameID_dict[right_img_id]
        left_image_dict = self.colmap_image_dataset[left_id]
        right_image_dict = self.colmap_image_dataset[right_id]

        pair_data = self.buildDataPair(left_image_dict, right_image_dict)

        pair_data.update(
            {
                "mkpts0_c": torch.from_numpy(left_kpts),
                "mkpts1_c": torch.from_numpy(right_kpts),
                "mkpts0_idx": torch.from_numpy(left_kpts_idx),
                "frame0_colmap_id": left_img_id,
                "frame1_colmap_id": right_img_id,
                # "image_base_dir": self.colmap_image_dataset.img_dir
            }
        )

        if self.warp_image1_by_homo:
            from .utils.sample_homo import sample_homography_sap

            h, w = pair_data["image1"].shape[-2:]
            homo_sampled = sample_homography_sap(h, w)

            pair_data.update({
                "homo_mat": torch.from_numpy(homo_sampled[None]) # B*3*3
            })

            # homo_sampled_normed = normalize_homography(
            #     torch.from_numpy(homo_sampled[None]).to(torch.float32), (h, w), (h, w)
            # )
            # # homo_warpped_image1 = transform.warp((pair_data['image1'].cpu().numpy() * 255).round().astype(np.int32)[0], homo_sampled)
            # homo_warpped_image1 = homography_warp(
            #     pair_data["image1"], torch.linalg.inv(homo_sampled_normed), (h, w)
            # )

            # # For debug:
            # from src.utils.plot_utils import plot_single_image, make_matching_plot

            # homo_warpped_image1 = (
            #     (homo_warpped_image1.cpu().numpy() * 255).round().astype(np.int32)[0]
            # )  # 1*H*W
            # fig = plot_single_image(homo_warpped_image1[0])
            # plt.savefig("homo_warpped_image.png")
            # plt.close()
            # original_image0 = (
            #     (pair_data["image0"].cpu().numpy() * 255).round().astype(np.int32)[0]
            # )  # 1*H*W
            # original_image1 = (
            #     (pair_data["image1"].cpu().numpy() * 255).round().astype(np.int32)[0]
            # )  # 1*H*W
            # fig = plot_single_image(original_image1[0])
            # plt.savefig("original_image.png")
            # plt.close()

            # # Plot matching:
            # norm_pixel_mat = normal_transform_pixel(h, w)
            # right_kpts_normed = (
            #     norm_pixel_mat[0].numpy()
            #     @ (
            #         np.concatenate(
            #             [right_kpts, np.ones((right_kpts.shape[0], 1))], axis=-1
            #         )
            #     ).T
            # ).astype(np.float32)
            # right_kpts_warpped = (
            #     norm_pixel_mat[0].inverse() @ homo_sampled_normed[0] @ right_kpts_normed
            # ).T  # N*3

            # right_kpts_warpped[:, :2] /= right_kpts_warpped[
            #     :, [2]
            # ]  # NOTE: Important! [:, 2] is not all 1!
            # right_kpts_warpped = right_kpts_warpped[:, :2]  # N*2
            # figure = make_matching_plot(
            #     original_image1[0],
            #     homo_warpped_image1[0],
            #     right_kpts,
            #     right_kpts_warpped,
            #     right_kpts,
            #     right_kpts_warpped,
            #     color=np.zeros((left_kpts.shape[0], 3)),
            #     text="",
            # )
            # plt.savefig("wrapped_match_pair.png")
            # plt.close()
            # homo_sampled = homo_sampled

        return pair_data

