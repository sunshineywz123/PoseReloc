from functools import partial
from pytorch3d import transforms
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

from submodules.DeepLM import Solve as SecondOrderSolve
from .first_order_solver import FirstOrderSolve

from .residual import pose_ba_residual, depth_residual
from ..utils.geometry_utils import *


class Optimizer(nn.Module):
    def __init__(self, optimization_dataset, configs=None):
        """
        Parameters:
        ----------------
        """
        super().__init__()
        # self.configs = configs
        # self.verbose = self.configs["verbose"]

        self.optimization_dataset = optimization_dataset
        self.colmap_frame_dict = optimization_dataset.colmap_frame_dict

        # DataLoading related
        self.num_workers = 12
        self.batch_size = 2000

        # self.solver_type = "SecondOrder"
        self.solver_type = "FirstOrder"
        self.residual_mode = (
            "feature_metric_error"  # ["feature_metric_error", "geometry_error"]
            # "geometry_error"  # ["feature_metric_error", "geometry_error"]
        )
        # self.residual_mode = "geometry_error" #["feature_metric_error", "geometry_error"]
        self.distance_loss_scale = 10
        # self.optimize_lr = {"depth": 1e-4, "pose": 1e-4, "BA": 5e-5}  # Baseline
        self.optimize_lr = {'depth': 1e-2, 'pose': 1e-2, 'BA': 1e-3} # Only useful for first order: current best solution
        self.image_i_f_scale = 2
        self.verbose = False

    @torch.enable_grad()
    def forward(self, optimization_procedures="BA"):
        """
        """
        # Data structure build from matched kpts
        aggregated_dict = {}

        logger.info("Loading data begin!")
        start_time = time.time()
        device = torch.device("cuda")

        if self.optimization_dataset.padding_data:
            dataloader = DataLoader(
                self.optimization_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

            for data_batch in tqdm(dataloader, total=len(dataloader)):
                # Make padding mask:
                batch_size, max_track_length = data_batch["intrinsic0"].shape[:2]
                mask = []
                for i in range(batch_size):
                    sub_mask = torch.full(
                        (max_track_length,), 0, dtype=torch.bool
                    )  # (max_track_length, )
                    sub_mask[: data_batch["n_query"][i]] = True
                    mask.append(sub_mask)
                mask = torch.stack(mask, dim=0)  # batch_size * max_track_length
                assert torch.sum(mask > 0) == torch.sum(
                    data_batch["n_query"]
                ), f"Bug exists here!"

                # Aggregate data
                for data_key, data_item in data_batch.items():
                    data_item_extracted = (
                        data_item[mask]
                        if data_item.shape[1] == max_track_length
                        else data_item.squeeze(1)
                    )
                    if data_key not in aggregated_dict:
                        aggregated_dict[data_key] = []
                    aggregated_dict[data_key].append(data_item_extracted.to(device))
        else:
            for id in tqdm(range(len(self.optimization_dataset))):
                data = self.optimization_dataset[id]
                for key, value in data.items():
                    if key not in aggregated_dict:
                        aggregated_dict[key] = []
                    aggregated_dict[key].append(value.to(device) if isinstance(value, torch.Tensor) else value)

        end_time = time.time()
        print(f"Consums: {end_time - start_time}")

        # Concat data and convert data format to double
        for key, value in aggregated_dict.items():
            if isinstance(value[0], torch.Tensor):
                aggregated_dict[key] = torch.cat(value).double()
            elif isinstance(value[0], np.ndarray):
                aggregated_dict[key] = np.concatenate(value)
            else:
                raise NotImplementedError

        # Struct depth index
        depth_indices = []
        for i in range(aggregated_dict["depth"].shape[0]):
            depth_indices.append(
                torch.full((int(aggregated_dict["n_query"][i]),), i, device=device)
            )
        depth_indices = torch.cat(depth_indices).long()  # N

        # Construct poses and poses indexs
        angleAxis_poses_dict = {
            k: convert_pose2angleAxis(v["initial_pose"])
            for k, v in self.colmap_frame_dict.items()
        }  # {colmap_frame_id: angleAxis 1*6}
        angleAxis_poses = torch.from_numpy(
            np.concatenate(list(angleAxis_poses_dict.values()))
        ).to(
            device
        )  # n_frames*6
        aggregated_dict.update({"angle_axis_to_world": angleAxis_poses})

        colmap_frame_ids = torch.from_numpy(
            np.array(list(angleAxis_poses_dict.keys()))
        ).to(
            device
        )  # n_frames

        # construct index project dict
        colmap_frame_id2_index = {
            k: v
            for k, v in zip(
                list(angleAxis_poses_dict.keys()), np.arange(len(angleAxis_poses_dict))
            )
        }
        # convert left colmap id to index
        left_pose_idxs = [
            colmap_frame_id2_index[colmap_frame_id]
            for colmap_frame_id in aggregated_dict["left_colmap_ids"]
            .cpu()
            .numpy()
            .tolist()
        ]
        right_pose_idxs = [
            colmap_frame_id2_index[colmap_frame_id]
            for colmap_frame_id in aggregated_dict["right_colmap_ids"]
            .cpu()
            .numpy()
            .tolist()
        ]
        left_pose_idxs = torch.tensor(left_pose_idxs).to(device).long()
        right_pose_idxs = torch.tensor(right_pose_idxs).to(device).long()

        # TODO: move img_i / img_f scale to global parameter
        aggregated_dict["scale0"] *= self.image_i_f_scale
        aggregated_dict["scale1"] *= self.image_i_f_scale

        # Prepare optimization data
        point_cloud_ids = aggregated_dict["point_cloud_id"].long()

        initial_residual = None
        final_residual = None
        for i, procedure in tqdm(
            enumerate(optimization_procedures), total=len(optimization_procedures)
        ):
            if procedure == "depth":
                logger.info(
                    f"Only depth optimization, optimize: {aggregated_dict['depth'].shape[0]}, depth parameters"
                )
                # only optimize depth, regard pose an constant
                aggregated_dict["left_pose_indexed"] = aggregated_dict[
                    "angle_axis_to_world"
                ][left_pose_idxs]
                aggregated_dict["right_pose_indexed"] = aggregated_dict[
                    "angle_axis_to_world"
                ][right_pose_idxs]
                variables_name = ["depth"]
                indices = [depth_indices]
                constants_name = [
                    "left_pose_indexed",
                    "right_pose_indexed",
                    "intrinsic0",
                    "intrinsic1",
                    "mkpts0_c",
                    "mkpts1_c",
                    "mkpts1_f",
                    "distance_map",
                    "scale0",
                    "scale1",
                ]
                residual_format = depth_residual
            elif procedure == "pose":
                logger.info(
                    f"Only optimize pose, optimize: {aggregated_dict['angle_axis_to_world'].shape[0]}*6 parameters"
                )
                # only optimize pose, regard depth as constant
                aggregated_dict["depth_indexed"] = aggregated_dict["depth"][
                    depth_indices
                ]

                variables_name = ["angle_axis_to_world"]
                indices = {0: [left_pose_idxs, right_pose_idxs]}  # only optimize pose
                constants_name = [
                    "depth_indexed",
                    "intrinsic0",
                    "intrinsic1",
                    "mkpts0_c",
                    "mkpts1_c",
                    "mkpts1_f",
                    "distance_map",
                    "scale0",
                    "scale1",
                ]
                residual_format = pose_ba_residual
            elif procedure == "BA":
                logger.info(
                    f"BA optimization, optimize: pose {aggregated_dict['angle_axis_to_world'].shape[0]}*6 parameters, depth {aggregated_dict['depth'].shape[0]} parameters"
                )
                variables_name = ["angle_axis_to_world", "depth"]
                indices = {0: [left_pose_idxs, right_pose_idxs], 1: [depth_indices]}
                constants_name = [
                    "intrinsic0",
                    "intrinsic1",
                    "mkpts0_c",
                    "mkpts1_c",
                    "mkpts1_f",
                    "distance_map",
                    "scale0",
                    "scale1",
                ]
                residual_format = pose_ba_residual
            else:
                raise NotImplementedError

            variables = [
                aggregated_dict[variable_name] for variable_name in variables_name
            ]  # depth L*1
            constants = [
                aggregated_dict[constanct_name] for constanct_name in constants_name
            ]

            # Refinement
            partial_paras = {"distance_loss_scale": self.distance_loss_scale, "mode": self.residual_mode}
            if self.solver_type == "SecondOrder":
                if residual_format is pose_ba_residual:
                    # pose optimization and ba is not implement for second order solver
                    raise NotImplementedError
                optimized_variables = SecondOrderSolve(
                    variables=variables,
                    constants=constants,
                    indices=indices,
                    fn=partial(residual_format, **partial_paras),
                )
            elif self.solver_type == "FirstOrder":
                optimization_cfgs = {
                    "lr": self.optimize_lr[procedure],
                    "optimizer": "Adam",
                    "max_steps": 1000,
                }

                optimized_variables, residual_inform = FirstOrderSolve(
                    variables=variables,
                    constants=constants,
                    indices=indices,
                    fn=partial(residual_format, **partial_paras),
                    optimization_cfgs=optimization_cfgs,
                    return_residual_inform=True,
                    verbose=self.verbose,
                )
                if i == 0:
                    initial_residual = residual_inform[0]
                final_residual = residual_inform[1]

            else:
                raise NotImplementedError

            # Update results
            for idx, variable_name in enumerate(variables_name):
                aggregated_dict[variable_name] = optimized_variables[idx]

        if initial_residual is not None and final_residual is not None:
            logger.info(
                "Initial residual: %E, Final residual: %E, decrease: %E, relative decrease: %f%%"
                % (
                    initial_residual,
                    final_residual,
                    initial_residual - final_residual,
                    (
                        (initial_residual - final_residual)
                        / (initial_residual + 1e-4)
                        * 100
                    ),
                )
            )

        # Convert angle axis pose to R,t
        R = transforms.so3_exponential_map(
            aggregated_dict["angle_axis_to_world"][:, :3]
        )  # n_frames*3*3
        t = aggregated_dict["angle_axis_to_world"][:, 3:6]  # n_frames*3

        return {
            "pose": [R.cpu().numpy(), t.cpu().numpy()],  # [n_frames*3*3, n_frames*3]
            "colmap_frame_ids": colmap_frame_ids.cpu().numpy(),  # [n_frames]
            "depth": aggregated_dict["depth"].cpu().numpy(),  # [n_point_clouds]
            "point_cloud_ids": point_cloud_ids.cpu().numpy(),  # [n_point_clouds]
        }
