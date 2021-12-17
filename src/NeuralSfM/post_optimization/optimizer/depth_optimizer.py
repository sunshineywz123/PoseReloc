from functools import partial
import torch
import torch.nn as nn

from submodules.DeepLM import Solve as SecondOrderSolve
from .first_order_solver import FirstOrderSolve

from .residual import residual


class DepthOptimizer(nn.Module):
    def __init__(self, depth_optimization_dataset, configs=None):
        """
        Parameters:
        ----------------
        """
        super().__init__()
        # self.configs = configs
        # self.verbose = self.configs["verbose"]

        self.depth_optimization_dataset = depth_optimization_dataset

        # self.solver_type = "SecondOrder"
        self.solver_type = "FirstOrder"
        self.image_i_f_scale = 2

    @torch.enable_grad()
    def forward(self):
        """
        """
        # Data structure build from matched kpts
        aggregated_dict = {}

        # TODO: multi-process data loading
        for id in range(len(self.depth_optimization_dataset)):
            data = self.depth_optimization_dataset[id]
            data_c = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }
            for key, value in data_c.items():
                if key not in aggregated_dict:
                    aggregated_dict[key] = []
                aggregated_dict[key].append(value)

        device = data_c["intrinsic0"].device

        # Struct index
        indices = []
        for i in range(len(aggregated_dict["depth"])):
            num_of_query_branchs = aggregated_dict["intrinsic1"][i].shape[0]
            indices.append(torch.full((num_of_query_branchs,), i, device=device))

        indices = torch.cat(indices).long()  # N
        for key, value in aggregated_dict.items():
            aggregated_dict[key] = torch.cat(value).double()

        # TODO: move img_i / img_f scale to global parameter
        aggregated_dict["scale0"] *= self.image_i_f_scale
        aggregated_dict["scale1"] *= self.image_i_f_scale

        # Prepare optimization data
        point_cloud_id = aggregated_dict.pop("point_cloud_id").long()

        variables_name = ["depth"]
        constants_name = [
            "intrinsic0",
            "intrinsic1",
            "angle_axis_relative",
            "mkpts0_c",
            "mkpts1_c",
            "mkpts1_f",
            "distance_map",
            "scale0",
            "scale1",
        ]
        variables = [
            aggregated_dict[variable_name] for variable_name in variables_name
        ]  # depth L*1
        indices = [indices]
        constants = [
            aggregated_dict[constanct_name] for constanct_name in constants_name
        ]

        # Refinement
        partial_paras = {"distance_loss_scale": 10, "mode": "feature_metric_error"}
        if self.solver_type == "SecondOrder":
            depth = SecondOrderSolve(
                variables=variables,
                constants=constants,
                indices=indices,
                fn=partial(residual, **partial_paras),
            )
        elif self.solver_type == "FirstOrder":
            optimization_cfgs = {
                "lr": 1e-2,
                "optimizer": "Adam",
                "max_steps": 1000,
            }

            depth = FirstOrderSolve(
                variables=variables,
                constants=constants,
                indices=indices,
                fn=partial(residual, **partial_paras),
                optimization_cfgs=optimization_cfgs,
            )
        else:
            raise NotImplementedError

        return {"depth": depth, "point_cloud_id": point_cloud_id}
