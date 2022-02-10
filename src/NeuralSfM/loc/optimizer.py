from functools import partial
from pytorch3d import transforms
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from tqdm import tqdm
import ray

# from submodules.DeepLM import Solve as SecondOrderSolve
from .first_order_solver import FirstOrderSolve
from .residual import pose_optimization_residual
from .utils.geometry_utils import *

class Optimizer(nn.Module):
    def __init__(self, cfgs):
        """
        Parameters:
        ----------------
        """
        super().__init__()
        self.residual_mode = cfgs["residual_mode"]
        self.solver_type = cfgs["solver_type"]
        self.distance_loss_scale = cfgs["distance_loss_scale"]
        # self.optimize_lr = {"depth": 1e-4, "pose": 1e-4, "BA": 5e-5}  # Baseline
        # First Order parameters:
        self.optimize_lr = cfgs["optimize_lr"]
        self.max_steps = cfgs['max_steps']

        self.image_i_f_scale = cfgs["image_i_f_scale"]
        self.verbose = cfgs["verbose"]

    @torch.enable_grad()
    def start_optimize(
        self,
        initial_pose,
        intrinsic,
        mkpts3d,
        mkpts2d_c,
        mkpts2d_f,
        scale,
        feature_3d,
        feature_2d_window,
        device,
        feature_distance_map_temperature=0.1,
        point_cloud_scale=1000
    ):
        """
        Parameters:
        ---------------
        initial_pose: [3*3, 3]
        intrinsic: [3*3]
        mkpts3d: [L*3]
        mkpts2d_c: [L*2]
        mkpts2d_f: [L*2]
        scale: [1*2]
        feature_3d: [L*1*D]
        feature_2d_window: [L*WW*D]
        device: torch.device
        point_cloud_scale: int
        """
        # Build feature distance map
        feature_distance_map = torch.linalg.norm(
            feature_3d - feature_2d_window, dim=-1, keepdim=True
        ).to(device)  # L*WW*1
        feature_distance_map /= feature_distance_map_temperature

        # Construct poses and poses indexs
        angleAxis_pose = convert_pose2angleAxis(initial_pose)
        pose_index = np.full((mkpts3d.shape[0],), 0)  # All zero index

        # Map and move to device
        (
            angleAxis_pose,
            pose_index,
            mkpts3d,
            mkpts2d_c,
            mkpts2d_f,
            intrinsic,
            scale,
        ) = map(
            lambda obj: torch.from_numpy(obj).to(device).to(torch.float32),
            [
                angleAxis_pose,
                pose_index,
                mkpts3d,
                mkpts2d_c,
                mkpts2d_f,
                intrinsic,
                scale,
            ],
        )

        scale *= self.image_i_f_scale
        mkpts3d *= point_cloud_scale
        angleAxis_pose[:,3:6] *= point_cloud_scale
        pose_index = pose_index.to(torch.long)

        # Refinement
        partial_paras = {
            "intrinsic": intrinsic,
            "scale": scale,
            "distance_loss_scale": self.distance_loss_scale,
            "mode": self.residual_mode,
        }
        if self.solver_type == "SecondOrder":
            raise NotImplementedError
            optimized_variables = SecondOrderSolve(
                variables=variables,
                constants=constants,
                indices=indices,
                fn=partial(residual_format, **partial_paras),
            )
        elif self.solver_type == "FirstOrder":
            optimization_cfgs = {
                "lr": self.optimize_lr,
                "optimizer": "Adam",
                "max_steps": self.max_steps,
            }

            optimized_variables, residual_inform = FirstOrderSolve(
                variables=[angleAxis_pose],
                constants=[mkpts3d, mkpts2d_c, mkpts2d_f, feature_distance_map],
                indices=[pose_index],
                fn=partial(pose_optimization_residual, **partial_paras),
                optimization_cfgs=optimization_cfgs,
                return_residual_inform=True,
                verbose=self.verbose,
            )

        # Convert angle axis pose to R,t
        R = transforms.so3_exponential_map(optimized_variables[0][:, :3]).squeeze(
            0
        )  # 3*3
        t = optimized_variables[0][:, 3:6].squeeze(0)  # n_frames*3

        # Rescale:
        t /= point_cloud_scale

        return np.concatenate([R.cpu().numpy(), t.cpu().numpy()[:, None]], axis=-1) # 3*4

    @ray.remote(num_cpus=1, num_gpus=1)  # release gpu after finishing
    def start_optimize_ray_wrapper(self):
        return self.start_optimize()
