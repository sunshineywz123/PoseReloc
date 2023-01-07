import numpy as np
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from loguru import logger
from .utils.geometry_utils import *
from .utils.io_utils import feature_load, feature_save


def feature_aggregation_and_update(
    colmap_image_dataset,
    fine_match_results_dict,
    feature_out_pth,
    image_lists,
    keypoints_update_method="colmap_updated_keypoints",
    aggregation_method="avg",
    verbose=True,
):
    # Update feature file
    feature_coarse_path = (
        osp.splitext(feature_out_pth)[0] + "_coarse" + osp.splitext(feature_out_pth)[1]
    )
    feature_dict_coarse = feature_load(feature_coarse_path, image_lists)
    feature_dict_fine = deepcopy(feature_dict_coarse)  # Store fine keypoints

    colmap_image_dataset = colmap_image_dataset
    colmap_3ds = colmap_image_dataset.colmap_3ds
    colmap_images = colmap_image_dataset.colmap_images
    point_cloud_assigned_imgID_kptsID_list = list(
        colmap_image_dataset.point_cloud_assigned_imgID_kptID.items()
    )
    fine_match_results_dict = fine_match_results_dict

    logger.info("Update feature and refined points begin...")
    if verbose:
        iter_obj = tqdm(range(len(point_cloud_assigned_imgID_kptsID_list)))
    else:
        iter_obj = range(len(point_cloud_assigned_imgID_kptsID_list))

    for index in iter_obj:
        point_cloudID, assigned_state = point_cloud_assigned_imgID_kptsID_list[index]
        assigned_colmap_frameID, assigned_keypoint_index = assigned_state

        image_ids = colmap_3ds[point_cloudID].image_ids.tolist()
        point2D_idxs = colmap_3ds[point_cloudID].point2D_idxs.tolist()

        pairs_dict = {}  # {"query_id-ref_id": keypoint_idx_in_ref_frame}
        query_kpt_idx = assigned_keypoint_index
        query_features_c = []
        query_features_f = []
        query_keypoints = []
        for image_id, kpt_id in zip(image_ids, point2D_idxs):
            if image_id != assigned_colmap_frameID:
                pair_name = "-".join([str(assigned_colmap_frameID), str(image_id)])
                ref_kpt_idx = kpt_id
                pairs_dict[pair_name] = kpt_id

                assert pair_name in fine_match_results_dict
                left_colmap_id, reight_colmap_id = pair_name.split("-")
                fine_match_results = fine_match_results_dict[pair_name]

                # Find corresponding features and keypoints
                index = np.argwhere(fine_match_results["mkpts0_idx"] == query_kpt_idx)
                assert len(index) == 1
                index = np.squeeze(index)

                feature_c0 = fine_match_results["feature_c0"][index]  # [dim]
                feature_c1 = fine_match_results["feature_c1"][index]  # [dim]

                feature0 = fine_match_results["feature0"][index]  # [dim]
                feature1 = fine_match_results["feature1"][index]  # [dim]

                if keypoints_update_method is not "colmap_updated_keypoints":
                    keypoints0 = fine_match_results["mkpts0_f"][index]
                    keypoints1 = fine_match_results["mkpts1_f"][index]
                    query_keypoints.append(keypoints0)

                query_features_c.append(feature_c0)
                query_features_f.append(feature0)  # Multiple feature0

                feat_c_dim = feature_c0.shape[0]
                feat_f_dim = feature0.shape[0]

                left_img_name = colmap_images[int(left_colmap_id)].name
                right_img_name = colmap_images[int(reight_colmap_id)].name

                # Reference feature dim check
                if (
                    feature_dict_coarse[right_img_name]["descriptors"].shape[0]
                    != feat_c_dim
                ):
                    num_kpts = feature_dict_coarse[right_img_name]["keypoints"].shape[
                        0
                    ]
                    feature_dict_coarse[right_img_name]["descriptors"] = np.zeros(
                        (feat_c_dim, num_kpts)
                    )
                if (
                    feature_dict_fine[right_img_name]["descriptors"].shape[0]
                    != feat_f_dim
                ):
                    num_kpts = feature_dict_fine[right_img_name]["keypoints"].shape[0]
                    feature_dict_fine[right_img_name]["descriptors"] = np.zeros(
                        (feat_f_dim, num_kpts)
                    )

                # Update reference feature and keypoints
                # NOTE: save fine fine feature to coarse feature dict temporary
                feature_dict_coarse[right_img_name]["descriptors"][
                    :, ref_kpt_idx
                ] = feature_c1

                feature_dict_fine[right_img_name]["descriptors"][
                    :, ref_kpt_idx
                ] = feature1

                if keypoints_update_method is not "colmap_updated_keypoints":
                    feature_dict_fine[right_img_name]["keypoints"][
                        ref_kpt_idx, :
                    ] = keypoints1

        # Update query feature:
        if feature_dict_coarse[left_img_name]["descriptors"].shape[0] != feat_c_dim:
            num_kpts = feature_dict_coarse[left_img_name]["keypoints"].shape[0]
            feature_dict_coarse[left_img_name]["descriptors"] = np.zeros(
                (feat_c_dim, num_kpts)
            )
        if feature_dict_fine[left_img_name]["descriptors"].shape[0] != feat_f_dim:
            num_kpts = feature_dict_fine[left_img_name]["keypoints"].shape[0]
            feature_dict_fine[left_img_name]["descriptors"] = np.zeros(
                (feat_f_dim, num_kpts)
            )

        query_features_c = np.stack(query_features_c, axis=0)  # N*dim
        query_features_f = np.stack(query_features_f, axis=0)  # N*dim

        if aggregation_method == "avg":
            query_features_c_agged = np.mean(
                query_features_c, axis=0, keepdims=False
            )  # [dim]
            query_features_f_agged = np.mean(
                query_features_f, axis=0, keepdims=False
            )  # [dim]
        else:
            raise NotImplementedError

        # Update query feature
        feature_dict_coarse[left_img_name]["descriptors"][
            : query_features_c_agged.shape[0], query_kpt_idx
        ] = query_features_c_agged

        feature_dict_fine[left_img_name]["descriptors"][
            : query_features_f_agged.shape[0], query_kpt_idx
        ] = query_features_f_agged

        feature_dict_coarse[left_img_name]["scores"][query_kpt_idx] = 0 # Just place-holder!
        feature_dict_fine[left_img_name]["scores"][query_kpt_idx] = 0

        if keypoints_update_method is not "colmap_updated_keypoints":
            query_keypoints = np.stack(query_keypoints, axis=0)  # N*2
            query_keypoints = np.mean(query_keypoints, axis=0, keepdims=False)  # [2]
            feature_dict_fine[left_img_name]["keypoints"][
                query_kpt_idx, :
            ] = query_keypoints
        else:
            for id, colmap_image in colmap_images.items():
                image_name = colmap_image.name
                updated_keypoints = colmap_image.xys
                origin_keypoints = feature_dict_fine[image_name]["keypoints"]
                if origin_keypoints.shape[0] != updated_keypoints.shape[0]:
                    logger.warning(f'origin_kpts: {origin_keypoints.shape[0]}, updated_kpts: {updated_keypoints.shape[0]}')
                feature_dict_fine[image_name]["keypoints"] = updated_keypoints

    # Save results (overwrite previous):
    feature_coarse_save_pth = (
        osp.splitext(feature_out_pth)[0] + "_coarse" + osp.splitext(feature_out_pth)[1]
    )
    feature_fine_save_pth = feature_out_pth
    feature_save(feature_dict_coarse, feature_coarse_save_pth)
    feature_save(feature_dict_fine, feature_fine_save_pth)
