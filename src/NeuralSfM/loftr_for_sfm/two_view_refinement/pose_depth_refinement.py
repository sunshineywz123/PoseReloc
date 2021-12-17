import torch.nn as nn
import torch
import math
from pytorch3d import transforms

from .two_view_pose_initialize import two_view_pose_initialize
from .threeD_points_initialize import (
    threeD_points_initialize_known_depth,
    threeD_points_initialize_opencv,
    threeD_points_initialize_DeepLM,
)
from .optimization_loss import (
    Two_view_featuremetric_error,
    Two_view_featuremetric_error_pose,
)
from .optimization_loss import (
    Two_view_reprojection_error,
    Two_view_reprojection_error_for_pose,
)
from .utils import (
    find_all_character,
    draw_heatmap_of_local_patch,
    sample_feature_from_featuremap,
    sample_feature_from_unfold_featuremap,
)
from src.utils.utils import compute_pose_error

from submodules.DeepLM import Solve
from .first_order_optimizer import FirstOrderSolve
from functools import partial
from loguru import logger
from time import time


class PoseDepthRefinement(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # prepare configs for DeepLM optimizer
        LM_configs = self.configs["refinement"]["lm"]
        self.first_order_pose_configs = {
            **self.configs["refinement"]["first_order"]["pose"],
            "optimizer": self.configs["refinement"]["first_order"]["optimizer"],
        }
        self.first_order_depth_configs = {
            **self.configs["refinement"]["first_order"]["depth"],
            "optimizer": self.configs["refinement"]["first_order"]["optimizer"],
        }
        self.first_order_ba_configs = {
            **self.configs["refinement"]["first_order"]["ba"],
            "optimizer": self.configs["refinement"]["first_order"]["optimizer"],
        }

        # reformat names of input paras to fit DeepLM paras: from name_part to namePart
        for key in list(LM_configs.keys()):
            if "_" in key:
                value = LM_configs.pop(key)
                indexs = find_all_character(key, "_")
                key = list(key)
                for index in indexs:
                    key[index + 1] = key[index + 1].upper()
                key = "".join(key).replace("_", "")
                LM_configs[key] = value

        self.LM_configs = LM_configs
        self.verbose = self.configs["refinement"]["verbose"]

    def forward(
        self,
        data,
        fine_preprocess=None,
        loftr_fine=None,
        pose_initialize_type="OpenCV",
        depth_initialize_type="DeepLM",
        refinement_type="both",
        **kwargs,
    ):
        """
        Parameters:
        -----------------
        kwargs: some configs to control direct method and feature based method
        {
            # direct method configs:
            "use_coarse_refinement" : False,
            "coarse_solver_type" : "second_order",
            "coarse_refine_type" : "pose_depth",
            "use_fine_refinement" : True,
            "fine_solver_type" : "first_order",
            "fine_refine_type" : "depth_pose"

            # feature based method configs:
            "feature_based_refine_type" : "pose_depth"
        }
        Update:
        ----------
        "pose_direct_refined" : [[R torch.tensor 3*3,t torch.tensor 3*1]]
        "pose_feature_based_refined" : [[R torch.tensor 3*3,t torch.tensor 3*1]]
        "depth_direct_refined" : [torch.tensor N]
        "depth_feature_based_refined" : [torch.tensor N]
        """
        # FIXME: not a good implementation, do not input fine_preprocess and loftr_fine, moreover recenter operation does't work for now
        self.fine_preprocess = fine_preprocess
        self.loftr_fine = loftr_fine

        self.K0, self.K1 = data["K0"], data["K1"]
        self.mbids, self.mkpts0, self.mkpts1 = (
            data["m_bids"],
            data["mkpts0_f"],
            data["mkpts1_f"],
        )
        self.device = data["mkpts0_f"].device

        # TODO: add hierarchical feature map
        # Get coarse level feature map and reshape to B*C*H*W
        self.N, self.L, self.S, self.C = (
            data["feat_c0"].size(0),
            data["feat_c0"].size(1),
            data["feat_c1"].size(1),
            data["feat_c0"].size(2),
        )
        self.feat_0 = [
            data["feat_c0"]
            .view(self.N, data["hw0_c"][0], data["hw0_c"][1], self.C)
            .permute(0, 3, 1, 2)
        ]
        self.feat_1 = [
            data["feat_c1"]
            .view(self.N, data["hw1_c"][0], data["hw0_c"][1], self.C)
            .permute(0, 3, 1, 2)
        ]

        # Get fine level unfold feature
        self.feature_fine0, self.feature_fine1 = (
            data["feat_f0_unfold"],
            data["feat_f1_unfold"],
        )
        self.M, WW, C = self.feature_fine0.shape
        self.W = int(math.sqrt(WW))

        # Get stereo pose initialize
        self.pose_initialize(data, pose_initialize_type=pose_initialize_type)

        # Depth(and point cloud) initialize
        self.depth_initialize(data, depth_initialize_type=depth_initialize_type)

        # Get initialization results
        self.pose_initial = data["initial_pose"]
        self.depth_initial = data["depth0_init"]
        self.point_cloud_bids = data["point_cloud_bids"]

        self.use_angle_axis = True
        self.use_depth_random_inital = False

        data.update(
            {
                "kpts_inlier_mask": [
                    pose_initial_[2] for pose_initial_ in self.pose_initial
                ]
            }
        )

        self.refinement_process(data, refinement_type=refinement_type, **kwargs)

    def pose_initialize(self, data, pose_initialize_type="OpenCV"):
        if pose_initialize_type == "OpenCV":
            two_view_pose_initialize(data, self.configs["pose_initial"])
        elif pose_initialize_type == "pose_prior":
            # data["pose_prior"]: B*4*4
            # Inlier mask problem
            assert "pose_prior" in data
            assert data["pose_prior"].shape[0] == data["K0"].shape[0]
            pose_prior = data["pose_prior"] # B*4*4
            initial_pose = []
            for bs in range(data["K0"].shape[0]):
                mask = data["m_bids"] == bs
                initial_pose.append(
                    [
                        pose_prior[bs][:3, :3],  # 3*3
                        pose_prior[bs][:3, 3:],  # 3*1
                        torch.ones(
                            (data["mkpts0_f"][mask].shape[0],), device=self.device
                        ).bool(),  # TODO: geometry verification, find inlier!
                    ]
                )
                data.update({"initial_pose": initial_pose})
        else:
            logger.info("Initialize pose as Identity rotation and zero translation")
            # Initialize as R: Identity matrix, t: zero matrix
            initial_pose = []
            for bs in range(data["K0"].shape[0]):
                mask = data["m_bids"] == bs
                initial_pose.append(
                    [
                        torch.eye(3, device=self.device),
                        torch.zeros((3, 1), device=self.device),
                        torch.ones(
                            (data["mkpts0_f"][mask].shape[0],), device=self.device
                        ).bool(),  # all true, [N]
                    ]
                )
            data.update({"initial_pose": initial_pose})

    def depth_initialize(self, data, depth_initialize_type="DeepLM"):
        if depth_initialize_type == "OpenCV":
            threeD_points_initialize_opencv(data, debug=True)
        elif depth_initialize_type == "DeepLM":
            threeD_points_initialize_DeepLM(
                data,
                self.LM_configs,
                method="reprojection_method",
                verbose=self.verbose,
            )
            # threeD_points_initialize_DeepLM(data, self.LM_configs, method='mid_point_method', verbose=self.verbose)
        elif depth_initialize_type == "both":
            # Used for debug
            start_time = time()
            threeD_points_initialize_opencv(data, debug=True)
            opencv_finish = time()
            threeD_points_initialize_DeepLM(
                data, self.LM_configs, method="mid_point_method", verbose=self.verbose
            )
            DeepLM_finish = time()
            logger.info(
                f"OpenCV triangulation use : {opencv_finish-start_time}s, DeepLM triangulation use: {DeepLM_finish - opencv_finish}s"
            )
        elif depth_initialize_type == "known_depth":
            # TODO: Debug here
            threeD_points_initialize_known_depth(data)
        else:
            raise NotImplementedError

    def refinement_process(self, data, refinement_type="both", **kwargs):
        # Pose refinement
        # NOTE: only used for debug TODO: remove feature_based_method and both, only direct_method left
        if refinement_type == "direct_method":
            data.update({"pose_direct_refined": [], "depth_direct_refined": []})
            self.direct_method_refine(data, **kwargs)
        elif refinement_type == "feature_based_method":
            data.update(
                {"pose_feature_based_refined": [], "depth_feature_based_refined": []}
            )
            self.feature_based_method_refine(data, **kwargs)
        elif refinement_type == "both":
            data.update(
                {
                    "pose_direct_refined": [],
                    "pose_feature_based_refined": [],
                    "depth_direct_refined": [],
                    "depth_feature_based_refined": [],
                }
            )
            self.direct_method_refine(data, **kwargs)
            self.feature_based_method_refine(data, **kwargs)
        elif refinement_type == "neither":
            # use initial pose as final pose, not refine!
            data.update(
                {
                    "pose_direct_refined": [
                        pose_initial_[:2] for pose_initial_ in self.pose_initial
                    ],
                    "pose_feature_based_refined": [
                        pose_initial_[:2] for pose_initial_ in self.pose_initial
                    ],
                    "depth_direct_refined": [data["depth0_init"]],
                    "depth_feature_based_refined": [data["depth0_init"]],
                }
            )
        else:
            raise NotImplementedError

    @torch.enable_grad()
    def direct_method_refine(
        self,
        data,
        debug=False,
        use_coarse_refinement=False,  # direct method type
        coarse_solver_type="second_order",
        coarse_refine_type="pose_depth",
        use_fine_refinement=True,
        fine_solver_type="first_order",
        fine_refine_type="depth_pose",
        **kwargs,
    ):
        for i in range(self.N):
            bid_mask = self.mbids == i
            point_cloud_bids_mask = self.point_cloud_bids == i

            mask = (
                bid_mask & self.pose_initial[i][2]
            )  # inlier mask and (optional) known depth mask

            depth_random_initial = torch.abs(
                torch.randn_like(
                    self.depth_initial[point_cloud_bids_mask][:, None],
                    dtype=torch.float64,
                    device=self.device,
                )
            )

            # Reformat initial depth and pose to satisify DeepLM's format requirement
            # NOTE: DeepLM requires that variables and constants' type is double(float64)
            depth = (
                self.depth_initial[point_cloud_bids_mask][:, None].double()
                if not self.use_depth_random_inital
                else depth_random_initial
            )  # L*1
            angle_axis = transforms.so3_log_map(
                self.pose_initial[i][0].unsqueeze(0)
            )  # bs = 1 convert R to angle-axis rotation vector for optimization
            T = torch.cat(
                [angle_axis, self.pose_initial[i][1].view(-1, 3)], dim=-1
            ).double()  # convert R[3,3], T[3,1] to 1*6

            # Get indices:
            depth_refine_indices = torch.arange(
                mask.float().sum(), dtype=torch.long, device=self.device
            )
            pose_refine_indices = torch.zeros(
                (self.mkpts0[mask].shape[0],), dtype=torch.long, device=self.device
            )

            # Get constants:
            mkpts0 = self.mkpts0[mask].double()
            mkpts1 = self.mkpts1[mask].double()

            # *************************************************
            # Pose and depth refinement use coarse level feature
            # TODO: remove in future
            use_coarse_feature_refinement = self.configs["refinement"][
                "use_coarse_feature_refinement"
            ]
            if use_coarse_refinement:
                # Sample feature of keypoints0
                feature0 = sample_feature_from_featuremap(
                    self.feat_0[0][i],
                    self.mkpts0[mask],
                    torch.tensor(data["hw0_i"], device=self.device) * data["scale0"][i],
                    norm_feature=self.configs["refinement"]["norm_feature"],
                )

                # Paras for fix error function
                partial_paras = {
                    "feature0": feature0,
                    "feature_map1": self.feat_1[0][i],
                    "K0": self.K0[i],
                    "K1": self.K1[i],
                    "img0_hw": (
                        torch.tensor(data["hw0_i"], device=self.device)
                        * data["scale0"][i]
                    ),
                    "img1_hw": (
                        torch.tensor(data["hw1_i"], device=self.device)
                        * data["scale1"][i]
                    ),
                    "norm_feature": self.configs["refinement"]["norm_feature"],
                    "residual_format": self.configs["refinement"]["residual_format"],
                    "use_angle_axis": self.use_angle_axis,
                }

                coarse_level_solver_type = self.configs["refinement"][
                    "coarse_level_solver_type"
                ]
                if coarse_solver_type == "first_order":
                    coarse_configs = {
                        "pose": [self.first_order_pose_configs],
                        "depth": [self.first_order_depth_configs],
                        "BA": [self.first_order_ba_configs],
                    }
                elif coarse_solver_type == "second_order":
                    # temporary
                    coarse_configs = [self.LM_configs]
                else:
                    raise NotImplementedError

                T, depth = self.direct_method_refinement_start(
                    T,
                    depth,
                    mkpts0,
                    mkpts1,
                    pose_refine_indices,
                    depth_refine_indices,
                    coarse_configs,
                    partial_paras,
                    solver_type=coarse_solver_type,
                    refine_type=coarse_refine_type,
                )

            # **********************************************
            # Pose and depth refinement use fine level unfold feature
            draw_heatmap = True  # debug paras, remove in future
            fine_patch_feature_size = self.configs["refinement"][
                "fine_patch_feature_size"
            ]  # whether use patch alignment
            use_fine_feature_refinement = self.configs["refinement"][
                "use_fine_feature_refinement"
            ]

            if self.M == 0:
                logger.warning(f"No matches found in coarse-level")
            elif use_fine_refinement:

                recenter_fine_feature = self.configs["refinement"][
                    "recenter_fine_feature"
                ]

                if recenter_fine_feature:
                    assert (
                        self.fine_preprocess is not None and self.loftr_fine is not None
                    )
                    assert (
                        fine_patch_feature_size is None
                    ), "Currently not implement patch alignment in recenter fine feature scenario"

                    offset0 = (
                        data["i_associated_kpts_local"][mask]
                        if "i_associated_kpts_local" in data
                        else None
                    )  # get offset of keypoints(in original image resolution)
                    feature0 = sample_feature_from_unfold_featuremap(
                        self.feature_fine0[mask],
                        offset0,
                        scale=data["hw0_i"][0] / data["hw0_f"][0],
                        norm_feature=self.configs["refinement"]["norm_feature"],
                    )

                    # used for debug, TODO: remove in future!
                    # draw heatmap for loftr fine grid feature before refinement
                    draw_heatmap_of_local_patch(
                        data["pair_names"][1][0],
                        feature0,
                        self.feature_fine1[mask],
                        data["mkpts1_c"][mask],
                        visual_type="distance",
                        save_dir="/data/hexingyi/NeuralSfM_visualize/heatmap_loftr_fine_feature",
                    ) if draw_heatmap else None

                    with torch.no_grad():
                        # feature map used for pose and depth refinement needs required_grad=False.
                        # NOTE: may change to torch.clone().detach()
                        # recenter fine unfold feature
                        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
                            data["feat_f0"],
                            data["feat_f1"],
                            data["feat_c0"],
                            data["feat_c1"],
                            data,
                            feats0=data["feats0"],
                            feats1=data["feats1"],
                            window_size=self.W,
                        )
                        feat_f0_unfold, feat_f1_unfold = self.loftr_fine(
                            feat_f0_unfold, feat_f1_unfold
                        )
                    self.feature_fine0, self.feature_fine1 = (
                        feat_f0_unfold,
                        feat_f1_unfold,
                    )
                    offset0 = None

                    # NOTE: mkpts1 should correspond to center of unfold feature, fine unfold features are recentered so use mkpts1_f
                    mkpts0 = data["mkpts0_f"][mask].double()
                    mkpts1 = data["mkpts1_f"][mask].double()
                else:
                    # get offset between keypoints0 and feature grid center point (in original image resolution)
                    offset0 = (
                        data["i_associated_kpts_local"][mask]
                        if "i_associated_kpts_local" in data
                        else None
                    )

                    # NOTE: mkpts1 should correspond to center of unfold feature, fine unfold feature are not recentered so use mkpts1_c
                    mkpts0 = data["mkpts0_f"][mask].double()
                    mkpts1 = data["mkpts1_c"][mask].double()

                # sample feature0 in advance incase of keypoints0 offset because of LoFTR-spp-grid
                feature0 = sample_feature_from_unfold_featuremap(
                    self.feature_fine0[mask],
                    offset0,
                    scale=data["hw0_i"][0] / data["hw0_f"][0],
                    norm_feature=self.configs["refinement"]["norm_feature"],
                    patch_feature_size=fine_patch_feature_size,
                )

                fine_level_solver_type = self.configs["refinement"][
                    "fine_level_solver_type"
                ]
                if fine_solver_type == "first_order":
                    fine_configs = {
                        "pose": [self.first_order_pose_configs],
                        "depth": [self.first_order_depth_configs],
                        "BA": [self.first_order_ba_configs],
                    }
                elif fine_solver_type == "second_order":
                    self.LM_configs["radius"] = self.configs[
                        "fine_level_lm_radius"
                    ]  # limit second order trust region to small number
                    # temporary
                    fine_configs = [self.LM_configs]
                elif fine_solver_type == "both":
                    # first order refine and then second order refine
                    self.LM_configs["radius"] = self.configs[
                        "fine_level_lm_radius"
                    ]  # limit second order trust region to small number
                    fine_configs = {
                        "pose": [self.first_order_pose_configs, self.LM_configs],
                        "depth": [self.first_order_depth_configs, self.LM_configs],
                        "BA": [self.first_order_ba_configs, self.LM_configs],
                    }
                else:
                    raise NotImplementedError

                draw_optimization_tragetory = True

                # wrap list to dict
                tragetory_dict = {"w_kpts0_list": [[]]}

                # for optimization loss function
                partial_paras = {
                    "feature0": feature0,
                    "feature_map1": self.feature_fine1[mask],
                    "K0": self.K0[i],
                    "K1": self.K1[i],
                    "scale": data["hw0_i"][0] / data["hw0_f"][0],
                    "norm_feature": self.configs["refinement"]["norm_feature"],
                    "residual_format": self.configs["refinement"]["residual_format"],
                    "use_angle_axis": self.use_angle_axis,
                    "use_fine_unfold_feature": True,
                    "return_reprojected_coord": draw_optimization_tragetory  # first order get tragetory_dict
                    and fine_solver_type != "second_order",
                    "enable_center_distance_loss": True,
                    "distance_loss_scale": 10,  # 80
                    "tragetory_dict": tragetory_dict  # second order get tragetory_dict from here
                    if draw_optimization_tragetory and fine_solver_type != "first_order"
                    else None,
                    "patch_size": fine_patch_feature_size,
                }

                T, depth = self.direct_method_refinement_start(
                    T,
                    depth,
                    mkpts0,
                    mkpts1,
                    pose_refine_indices,
                    depth_refine_indices,
                    fine_configs,
                    partial_paras,
                    solver_type=fine_solver_type,
                    refine_type=fine_refine_type,
                    tragetory_dict=tragetory_dict  # first order get tragetory_dict from here
                    if draw_optimization_tragetory
                    and fine_solver_type != "second_order"
                    else None,
                )

                # draw heatmap for recentered feature grid
                draw_heatmap_of_local_patch(
                    data["pair_names"][1][0],
                    feature0,
                    self.feature_fine1[mask],
                    data["mkpts1_f"][mask]
                    if recenter_fine_feature
                    else data["mkpts1_c"][mask],
                    reprojected_kpts_list=tragetory_dict["w_kpts0_list"]
                    if draw_optimization_tragetory
                    else None,
                    matched_kpts=data["mkpts1_f"][mask],
                    scale=data["hw0_i"][0] / data["hw0_f"][0],
                    visual_type="distance",
                    save_dir="/data/hexingyi/NeuralSfM_visualize/heatmap_used_for_refine",
                ) if draw_heatmap else None
            else:
                pass

            R_refined = transforms.so3_exponential_map(T[:, :3])  # 1*3*3
            t_refined = T[:, 3:6].unsqueeze(-1)  # 1*3*1
            T_refined = torch.cat([R_refined, t_refined], dim=-1)  # 1*3*4

            # NOTE: support bs=1 only
            if debug:
                dump_results_for_visualize(
                    data,
                    depth,
                    T_refined,
                    self.mkpts0[mask],
                    self.K0[i],
                    self.K1[i],
                    label="direct_method",
                    T_gt=data["T_0to1"][i],
                )
            data["pose_direct_refined"].append(
                [R_refined.squeeze(), t_refined.squeeze(0)]
            )
            data["depth_direct_refined"].append(depth.squeeze(1))

    @torch.enable_grad()
    def feature_based_method_refine(
        self, data, debug=False, feature_based_refine_type="pose_depth", **kwargs
    ):
        number_iteration = self.configs["refinement"]["refinement_iteration_number"]

        if feature_based_refine_type == "pose_depth":
            process_list = ["pose", "depth"] * number_iteration
            process_list += ["BA"]
        elif feature_based_refine_type == "depth_pose":
            process_list = ["depth", "pose"] * number_iteration
            process_list += ["BA"]
        elif feature_based_refine_type == "only_depth":
            process_list = ["depth"] * number_iteration
        elif feature_based_refine_type == "only_pose":
            process_list = ["pose"] * number_iteration
        elif feature_based_refine_type == "only_BA":
            process_list = ["BA"]
        else:
            raise NotImplementedError

        for i in range(self.N):
            bid_mask = self.mbids == i
            point_cloud_bids_mask = self.point_cloud_bids == i

            mask = bid_mask & self.pose_initial[i][2]

            depth_random_initial = torch.abs(
                torch.randn_like(
                    self.depth_initial[point_cloud_bids_mask][:, None],
                    dtype=torch.float64,
                    device=self.device,
                )
            )

            # Get initial depth and pose
            depth = (
                self.depth_initial[point_cloud_bids_mask][:, None].double()
                if not self.use_depth_random_inital
                else depth_random_initial
            )  # L*1
            angle_axis = transforms.so3_log_map(
                self.pose_initial[i][0].unsqueeze(0)
            )  # bs = 1 convert R to angle-axis rotation vector for optimization
            T = torch.cat(
                [angle_axis, self.pose_initial[i][1].view(-1, 3)], dim=-1
            ).double()  # convert R[3,3], T[3,1] to 1*6

            # Get indices:
            depth_refine_indices = torch.arange(
                mask.float().sum(), dtype=torch.long, device=self.device
            )
            pose_refine_indices = torch.zeros(
                (self.mkpts0[mask].shape[0],), dtype=torch.long, device=self.device
            )

            # paras for fix error function
            partial_paras = {
                "K0": self.K0[i],
                "K1": self.K1[i],
                "use_angle_axis": self.use_angle_axis,
            }

            # Start refinement
            for j, type in enumerate(process_list):
                if type == "pose":
                    if self.verbose:
                        logger.info(
                            f"Iteration:{j//2 if 'depth' in process_list else j},refine pose start......"
                        )
                    T = Solve(
                        variables=[T],  # pose
                        constants=[
                            depth,
                            self.mkpts0[mask].double(),
                            self.mkpts1[mask].double(),
                        ],  # depth ,kpts0
                        indices=[pose_refine_indices],
                        fn=partial(
                            Two_view_reprojection_error_for_pose, **partial_paras
                        ),
                        optimization_cfgs=self.LM_configs,
                        verbose=self.verbose,
                    )

                elif type == "depth":
                    # Refine depth with pose fixed
                    if self.verbose:
                        logger.info(
                            f"Iteration:{j//2 if 'pose' in process_list else i}, refine depth start......"
                        )
                    depth = Solve(
                        variables=[depth],
                        # variables=[depth_random_initial],
                        constants=[
                            T.expand(
                                self.mkpts0[mask].shape[0], -1, -1
                            ).double(),  # pose: from 1*3*4 to L*3*4
                            self.mkpts0[mask].double(),  # kpts0
                            self.mkpts1[mask].double(),  # kpts1
                        ],
                        indices=[depth_refine_indices],
                        fn=partial(Two_view_reprojection_error, **partial_paras),
                        optimization_cfgs=self.LM_configs,
                        verbose=self.configs["refinement"]["verbose"],
                    )

                elif type == "BA":
                    # refine pose and depth simultaneously:
                    if self.verbose:
                        logger.info(f"BA start......")
                    depth, T = Solve(
                        variables=[depth, T,],  # depth, pose
                        constants=[
                            self.mkpts0[mask].double(),
                            self.mkpts1[mask].double(),
                        ],  # kpts0 and kpts1
                        indices=[depth_refine_indices, pose_refine_indices],
                        fn=partial(Two_view_reprojection_error, **partial_paras),
                        optimization_cfgs=self.LM_configs,
                        verbose=self.configs["refinement"]["verbose"],
                    )
                else:
                    raise NotImplementedError

            R_refined = transforms.axis_angle_to_matrix(T[:, :3])
            t_refined = T[:, 3:6].unsqueeze(-1)
            T_refined = torch.cat([R_refined, t_refined], dim=-1)  # 1*3*4

            if debug:
                # NOTE: support bs=1 only
                dump_results_for_visualize(
                    data,
                    depth,
                    T_refined,
                    self.mkpts0[mask],
                    self.K0[i],
                    self.K1[i],
                    label="feature_based_method",
                    T_gt=data["T_0to1"][i],
                )
            data["pose_feature_based_refined"].append(
                [R_refined.squeeze(), t_refined.squeeze(0)]
            )
            data["depth_feature_based_refined"].append(depth.squeeze(1))

    def direct_method_refinement_start(
        self,
        T,
        depth,
        mkpts0,
        mkpts1,
        pose_refine_indices,
        depth_refine_indices,
        optimizer_configs,
        partial_paras,
        solver_type="second_order",
        refine_type="pose_depth",
        **kwargs,
    ):
        """
        Direct method pose and depth refinement, 
        alternatively refine depth and pose for setted number and process BA finally 

        Parameters:
        -------------
        T : torch.tensor float64 1*6
            two view relative pose
        depth : torch.tensor float64 L*1
        mkpts0 : torch.tensor float64 L*2
        mkpts1 : torch.tensor float64 L*2
        pose_refine_indices : torch.tensor L
        depth_refine_indices : torch.tensor L
        partial_paras : Dict
            parameters to fix part of DeepLM loss function
            note that tensor type in partial_paras is float32
        LM_configs : Dict{'pose' : Dict, 'depth' : Dict, 'BA' : Dict}
        solver_type : str
            choice : ['first_order', 'second_order', 'both']
        refine_type : str
            choice : ['pose_depth', 'depth_pose', 'only_depth', 'only_pose']
        kargs : 
            tragetory_dict : Dict{'w_kpts0_list' : List[torch.tensor L*2] n}
                only available when solver_type == 'one_order'

        Returns:
        ----------
        refined T : torch.tensor float64 1*6
        refined depth : torch.tensor float64 L*1

        """
        number_iteration = self.configs["refinement"]["refinement_iteration_number"]

        # use identical configs for pose & depth & BA scenario
        if "pose" not in optimizer_configs:
            optimizer_configs = {
                "pose": optimizer_configs,
                "depth": optimizer_configs,
                "BA": optimizer_configs,
            }

        if solver_type == "first_order":
            solvers = [FirstOrderSolve]
        elif solver_type == "second_order":
            solvers = [Solve]
        elif solver_type == "both":
            solvers = [FirstOrderSolve, Solve]
        else:
            raise NotImplementedError

        if refine_type == "pose_depth":
            process_list = ["pose", "depth"] * number_iteration
            process_list += ["BA"]
        elif refine_type == "depth_pose":
            process_list = ["depth", "pose"] * number_iteration
            process_list += ["BA"]
        elif refine_type == "only_depth":
            process_list = ["depth"] * number_iteration
        elif refine_type == "only_pose":
            process_list = ["pose"] * number_iteration
        elif refine_type == "only_BA":
            process_list = ["BA"]
        else:
            raise NotImplementedError

        # Start refinement
        for i, type in enumerate(process_list):

            if type == "pose":
                # Refine pose with depth fixed
                if self.verbose:
                    logger.info(
                        f"Iteration:{i//2 if 'depth' in process_list else i},refine pose start......"
                    )
                constants = [
                    depth,
                    mkpts0,
                    mkpts1,
                ]
                for j, solver in enumerate(solvers):
                    T = solver(
                        variables=[T],  # pose
                        constants=constants,  # depth ,kpts0, kpts1
                        indices=[pose_refine_indices],
                        fn=partial(Two_view_featuremetric_error_pose, **partial_paras),
                        optimization_cfgs=optimizer_configs["pose"][j],
                        verbose=self.verbose,
                        **kwargs,
                    )
            elif type == "depth":
                # Refine depth with pose fixed
                if self.verbose:
                    logger.info(
                        f"Iteration:{i//2 if 'pose' in process_list else i}, refine depth start......"
                    )
                constants = [
                    T.expand(mkpts0.shape[0], -1),  # pose: from 1*6 to L*6
                    mkpts0,  # kpts0
                    mkpts1,  # kpts1
                ]
                for j, solver in enumerate(solvers):
                    depth = solver(
                        variables=[depth],
                        # variables=[depth_random_initial],
                        constants=constants,
                        indices=[depth_refine_indices],
                        fn=partial(Two_view_featuremetric_error, **partial_paras),
                        optimization_cfgs=optimizer_configs["depth"][j],
                        verbose=self.configs["refinement"]["verbose"],
                        **kwargs,
                    )
            elif type == "BA":
                # refine pose and depth simultaneously (BA):
                if self.verbose:
                    logger.info(f"BA start......")
                constants = [mkpts0, mkpts1]  # kpts0, kpts1
                for j, solver in enumerate(solvers):
                    depth, T = solver(
                        variables=[depth, T,],  # depth, pose
                        # variables=[depth_random_initial],
                        constants=constants,
                        indices=[depth_refine_indices, pose_refine_indices],
                        fn=partial(Two_view_featuremetric_error, **partial_paras),
                        optimization_cfgs=optimizer_configs["BA"][j],
                        verbose=self.configs["refinement"]["verbose"],
                        **kwargs,
                    )
            else:
                raise NotImplementedError

            if "decay" in optimizer_configs[type][0]:
                # only first order optimizer has decay parameter
                optimizer_configs[type][0]["lr"] *= optimizer_configs[type][0]["decay"]

        return T, depth


@torch.no_grad()
def dump_results_for_visualize(
    data, depth_of_kpts0, pose, kpts0, K0, K1, label="direct_method", T_gt=None
):
    """
    Parameters:
    -----------
    pose: torch.tensor 3*4 | 1*3*4
    """
    assert label in ["direct_method", "feature_based_method"], "invaild lable name!"
    pose = pose.squeeze(0) if len(pose.shape) == 3 else pose
    depth, pose = map(lambda a: a.float(), [depth_of_kpts0, pose])

    kpts0_h = (
        torch.cat([kpts0, torch.ones((kpts0.shape[0], 1), device=kpts0.device)], dim=-1)
        * depth
    )  # (N, 3)
    point_cloud = K0.inverse() @ kpts0_h.transpose(1, 0)  # (3,L)

    threeD_coords_in_camera1 = pose[:, :3] @ point_cloud + pose[:, 3:]

    reprojection_coord_h = (K1 @ threeD_coords_in_camera1).transpose(1, 0)
    reprojection_coord = reprojection_coord_h[:, :2] / (
        reprojection_coord_h[:, [2]] + 1e-4
    )

    if T_gt is not None:
        t_error, R_error = compute_pose_error(
            T_gt.cpu().numpy(), pose[:, :3].cpu().numpy(), pose[:, 3].cpu().numpy()
        )
        data.update(
            {
                "_".join(["R_error_by", label]): R_error,
                "_".join(["t_error_by", label]): t_error,
            }
        )

    data.update(
        {
            "_".join(["threeD_refined_by", label]): point_cloud.transpose(1, 0),
            "_".join(["image1_reprojected_keypoints_by", label]): reprojection_coord,
        }
    )
