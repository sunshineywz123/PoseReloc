from time import time
import torch
import torch.optim as optim
from loguru import logger


def FirstOrderSolve(
    variables,
    constants,
    indices,
    fn,
    optimization_cfgs=None,
    verbose=False,
    tragetory_dict=None,
    return_residual_inform=False,
):
    """
    Parameters:
    ------------
    tragetory_dict: Dict{'w_kpts0_list':[[torch.tensor L*2][]]}
    """
    # Make confidance:
    initial_confidance = 50
    confidance = torch.full(
        (constants[0].shape[0],), fill_value=initial_confidance, requires_grad=False
    )
    # confidance = None

    max_steps = optimization_cfgs["max_steps"]

    # variable, constants and optimizer initialization
    variables = [torch.nn.Parameter(v) for v in variables]

    optimizer_type = optimization_cfgs["optimizer"]
    if "lr" in optimization_cfgs:
        # pose and depth refinement scenario
        lr = optimization_cfgs["lr"]
        # optimizer = optim.AdamW(variables, lr=lr)
        if optimizer_type == "Adam":
            optimizer = optim.Adam(variables, lr=lr)
        elif optimizer_type in ["SGD", "RMSprop"]:
            if optimizer_type == "SGD":
                optimizerBuilder = optim.SGD
            elif optimizer_type == "RMSprop":
                optimizerBuilder = optim.RMSprop
            else:
                raise NotImplementedError
            optimizer = optimizerBuilder(
                variables,
                lr=lr,
                momentum=optimization_cfgs["momentum"],
                weight_decay=optimization_cfgs["weight_decay"],
            )
        else:
            raise NotImplementedError
    else:
        # BA scenario, set lr for pose and depth respectively
        lr_list = [optimization_cfgs["depth_lr"], optimization_cfgs["pose_lr"]]
        if optimizer_type == "Adam":
            optimizer = optim.Adam(
                [
                    {"params": variable, "lr": lr_list[i]}
                    for i, variable in enumerate(variables)
                ]
            )
        elif optimizer_type in ["SGD", "RMSprop"]:
            if optimizer_type == "SGD":
                optimizerBuilder = optim.SGD
            elif optimizer_type == "RMSprop":
                optimizerBuilder = optim.RMSprop
            else:
                raise NotImplementedError
            optimizer = optimizerBuilder(
                [
                    {
                        "params": variable,
                        "lr": lr_list[i],
                        "momentum": optimization_cfgs["momentum"],
                        "weight_decay": optimization_cfgs["weight_decay"],
                    }
                    for i, variable in enumerate(variables)
                ]
            )
        else:
            raise NotImplementedError
    constantsPar = constants

    # NOTE: used for debug and visualize
    # record every refine steps' reprojection coordinates
    w_kpts0_list = []

    start_time = time()
    for i in range(max_steps):
        current_iteration_start_time = time()

        variables_expanded = []
        if isinstance(indices, list):
            # indices: [indexs: n]
            for variable_id, variable_index in enumerate(indices):
                variable_expanded = variables[variable_id][variable_index]
                variables_expanded.append(variable_expanded)
        elif isinstance(indices, dict):
            # indices: {variable_id: [indexs: n]}
            for variable_id, variable_indexs in indices.items():
                assert isinstance(variable_indexs, list)
                variable_expanded = [
                    variables[variable_id][variable_index]
                    for variable_index in variable_indexs
                ]

                # NOTE: Left pose should not BP gradients!
                if variable_id == 0:
                    variable_expanded[0] = variable_expanded[
                        0
                    ].detach()  # Left Pose Scenario

                variables_expanded += variable_expanded
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        try:
            results, confidance = fn(
                *variables_expanded,
                *constantsPar,
                confidance=confidance,
                verbose=verbose,
                marker=False,
                marker_return=True
            )
        except:
            results = fn(
                *variables_expanded,
                *constantsPar,
                confidance=confidance,
                verbose=verbose,
                marker=False,
                marker_return=True
            )
        if isinstance(results, torch.Tensor):
            residuals = results
        else:
            # NOTE: only used for debug, remove in future
            residuals, w_kpts0 = results
            w_kpts0_list.append(w_kpts0.clone().detach())
        # residuals = residuals.view(residuals.shape[0], -1)

        # l = torch.sum(residuals)
        l = torch.sum(0.5 * residuals * residuals)
        l.backward()
        optimizer.step()

        current_step_residual = l.clone().detach()
        current_time = time()
        # torch.cuda.synchronize()
        if i == 0:
            initial_residual = current_step_residual
            last_residual = initial_residual
            print(
                "Start one order optimization, residual = %E, total_time = %f ms"
                % (initial_residual, (current_time - start_time) * 1000)
            ) if verbose else None

        else:
            relative_decrease_rate = (
                last_residual - current_step_residual
            ) / last_residual
            print(
                "iter = %d, residual = %E, relative decrease percent= %f%%, current_iter_time = %f ms, total time = %f ms, %d residuals filtered"
                % (
                    i - 1,
                    current_step_residual,
                    relative_decrease_rate * 100,
                    (current_time - current_iteration_start_time) * 1000,
                    (current_time - start_time) * 1000,
                    torch.sum(confidance <= 0) if confidance is not None else 0,
                )
            ) if verbose else None
            last_residual = current_step_residual
            if relative_decrease_rate < 0.0001 and i > max_steps * 0.2:
                print("early stop!") if verbose else None
                break

    start_time = time()
    with torch.no_grad():
        results, confidance = fn(
            *variables_expanded,
            *constantsPar,
            confidance=confidance,
            verbose=verbose,
            marker=False,
            marker_return=True
        )
        if isinstance(results, torch.Tensor):
            residuals = results
        else:
            # NOTE: only used for debug, remove in future
            residuals, w_kpts0 = results
            # w_kpts0_list.append(w_kpts0.clone().detach())
        # finial_residual = torch.sum(residuals)
        finial_residual = torch.sum(0.5 * residuals * residuals)
    # torch.cuda.synchronize()
    print(
        "First order optimizer initial residual = %E , finial residual = %E, decrease = %E, relative decrease percent = %f%%, %d residuals filtered"
        % (
            initial_residual,
            finial_residual,
            initial_residual - finial_residual,
            ((initial_residual - finial_residual) / (initial_residual + 1e-4) * 100),
            torch.sum(confidance <= 0) if confidance is not None else 0,
        )
    ) if verbose else None
    if tragetory_dict is not None:
        w_kpts0_global_list = tragetory_dict["w_kpts0_list"]
        poped_list = w_kpts0_global_list.pop()
        w_kpts0_global_list += (
            [w_kpts0_list, [],]
            if len(poped_list) == 0
            else [poped_list, w_kpts0_list, []]
        )  # NOTE: [] is used for second order tragetory kpts update
        if len(w_kpts0_list) == 0:
            logger.warning(
                "Want to get reprojection points from loss function, however get nothing! please check whether set 'return_reprojected_coord' = True in loss function"
            )
        tragetory_dict.update({"w_kpts0_list": w_kpts0_global_list})
    variables = [variable.detach() for variable in variables]

    if return_residual_inform:
        return variables, [initial_residual, finial_residual]
    else:
        return variables
