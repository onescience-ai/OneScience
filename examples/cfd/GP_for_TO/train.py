import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
from gpytorch.settings import cholesky_jitter
from tqdm import tqdm

from onescience.utils.GP_TO import (
    plot_density_and_velocity_fields,
    plot_loss_history,
    plot_predictions_and_residuals,
)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 15
plt.rcParams["figure.dpi"] = 150


checkpoints = [1000, 10000, 20000, 30000, 40000, 50000]
lambdaa = 0.1

# Determine volume loss based on title
gamma_values = {
    "rugby": 0.9,
    "pipebend": 0.08 * torch.pi,
    "doublepipe": 1 / 3,
    "diffuser": 0.5,
}


def modified_sigmoid(x, alpha=12.0):
    """Compute the modified sigmoid function."""
    return 1 / (1 + torch.exp(-alpha * (x - 0.5)))


def compute_dynamic_weights_ic(pde_loss, bc_loss, ic_loss, model):
    """
    Compute dynamic weights for loss functions based on gradients.
    """
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    def compute_gradients(loss, scaling_factor):
        gradients = torch.autograd.grad(
            scaling_factor * loss,
            params_to_update,
            retain_graph=True,
            allow_unused=True,
        )
        values = [p.reshape(-1).cpu().tolist() for p in gradients if p is not None]
        return torch.abs(torch.tensor([v for val in values for v in val]))

    delta_pde = compute_gradients(pde_loss, 1.0)
    delta_bc = compute_gradients(bc_loss, model.alpha)
    delta_ic = compute_gradients(ic_loss, model.beta)

    temp_bc = torch.max(delta_pde) / torch.mean(delta_bc)
    temp_ic = torch.max(delta_pde) / torch.mean(delta_ic)

    return (
        (1.0 - lambdaa) * model.alpha + lambdaa * temp_bc,
        (1.0 - lambdaa) * model.beta + lambdaa * temp_ic,
    )


def compute_fvm_residuals_higher_order(u, v, p, f_x, f_y, nu, dx, dy):
    # Check the device of the input tensors and move other tensors to the same device
    device = u.device

    # Extend the fields with ghost cells to handle boundary conditions
    u_ext = torch.zeros((u.shape[0] + 4, u.shape[1] + 4), device=device)
    v_ext = torch.zeros((v.shape[0] + 4, v.shape[1] + 4), device=device)
    p_ext = torch.zeros((p.shape[0] + 4, p.shape[1] + 4), device=device)

    # Copy the interior
    u_ext[2:-2, 2:-2] = u
    v_ext[2:-2, 2:-2] = v
    p_ext[2:-2, 2:-2] = p

    # Apply ghost cells for no-slip boundary conditions
    u_ext[:2, :] = u_ext[2:4, :]
    u_ext[-2:, :] = u_ext[-4:-2, :]
    u_ext[:, :2] = u_ext[:, 2:4]
    u_ext[:, -2:] = u_ext[:, -4:-2]

    v_ext[:2, :] = v_ext[2:4, :]
    v_ext[-2:, :] = v_ext[-4:-2, :]
    v_ext[:, :2] = v_ext[:, 2:4]
    v_ext[:, -2:] = v_ext[:, -4:-2]

    # Laplacian of u and v (fourth-order central differences for second derivatives)
    u_xx = (
        -u_ext[4:, 2:-2]
        + 16 * u_ext[3:-1, 2:-2]
        - 30 * u_ext[2:-2, 2:-2]
        + 16 * u_ext[1:-3, 2:-2]
        - u_ext[:-4, 2:-2]
    ) / (12 * dx**2)
    u_yy = (
        -u_ext[2:-2, 4:]
        + 16 * u_ext[2:-2, 3:-1]
        - 30 * u_ext[2:-2, 2:-2]
        + 16 * u_ext[2:-2, 1:-3]
        - u_ext[2:-2, :-4]
    ) / (12 * dy**2)
    laplacian_u = u_xx + u_yy

    v_xx = (
        -v_ext[4:, 2:-2]
        + 16 * v_ext[3:-1, 2:-2]
        - 30 * v_ext[2:-2, 2:-2]
        + 16 * v_ext[1:-3, 2:-2]
        - v_ext[:-4, 2:-2]
    ) / (12 * dx**2)
    v_yy = (
        -v_ext[2:-2, 4:]
        + 16 * v_ext[2:-2, 3:-1]
        - 30 * v_ext[2:-2, 2:-2]
        + 16 * v_ext[2:-2, 1:-3]
        - v_ext[2:-2, :-4]
    ) / (12 * dy**2)
    laplacian_v = v_xx + v_yy

    # Gradients of p using fourth-order central difference
    grad_p_x = (
        -p_ext[4:, 2:-2]
        + 8 * p_ext[3:-1, 2:-2]
        - 8 * p_ext[1:-3, 2:-2]
        + p_ext[:-4, 2:-2]
    ) / (12 * dx)
    grad_p_y = (
        -p_ext[2:-2, 4:]
        + 8 * p_ext[2:-2, 3:-1]
        - 8 * p_ext[2:-2, 1:-3]
        + p_ext[2:-2, :-4]
    ) / (12 * dy)

    # Momentum equation residuals
    residual_u = -nu * laplacian_u + grad_p_x + f_x
    residual_v = -nu * laplacian_v + grad_p_y + f_y

    # Mass conservation (divergence of velocity) using central differences
    div_u = (u_ext[2:-2, 2:-2] - u_ext[1:-3, 2:-2]) / dx
    div_v = (v_ext[2:-2, 2:-2] - v_ext[2:-2, 1:-3]) / dy

    u_y = (u_ext[2:-2, 2:-2] - u_ext[2:-2, 1:-3]) / dy
    v_x = (v_ext[2:-2, 2:-2] - v_ext[1:-3, 2:-2]) / dx
    residual_mass = div_u + div_v

    return (
        residual_u.reshape(-1),
        residual_v.reshape(-1),
        residual_mass.reshape(-1),
        div_u.reshape(-1),
        div_v.reshape(-1),
        u_y.reshape(-1),
        v_x.reshape(-1),
    )


def compute_autograd_derivatives(model, z_column, collocation_x, grad_order=1):
    """
    Compute first and second-order derivatives using PyTorch autograd.
    :param model: Model to train
    :param z_column: Column of `z_all` for differentiation
    :param collocation_x: Input coordinates
    :param grad_order: Order of gradients (1 for first, 2 for second)
    :return: First and second-order derivatives as tensors
    """
    model.train()
    grad_1 = torch.autograd.grad(
        z_column, collocation_x, torch.ones_like(z_column), create_graph=True
    )[0]
    if grad_order > 1:
        grad_2_x = torch.autograd.grad(
            grad_1[:, 0],
            collocation_x,
            torch.ones_like(grad_1[:, 0]),
            create_graph=True,
        )[0][:, 0]
        grad_2_y = torch.autograd.grad(
            grad_1[:, 1],
            collocation_x,
            torch.ones_like(grad_1[:, 1]),
            create_graph=True,
        )[0][:, 1]
        return grad_1[:, 0], grad_1[:, 1], grad_2_x, grad_2_y
    return grad_1[:, 0], grad_1[:, 1], None, None


w_1, w_2, w_3, w_4, w_5 = 0.01, 0.01, 1e2, 1e5, 1e4


def loss_volume(y, gamma=0.5):
    """
    Compute the volume loss as the squared difference between the mean of y and gamma.
    """
    mean_y = torch.mean(y)
    return torch.square(mean_y - gamma)


def dissipated_power(u, u_x, v_y, u_y, v_x):
    """
    Compute the total dissipated power based on the input velocity gradients and displacements.
    """
    # First part of the dissipated power
    p1 = (u_x**2 + v_y**2 + u_y**2 + v_x**2).sum(dim=1, keepdim=True)

    # Second part of the dissipated power
    u2 = (u[:, :2] ** 2).sum(dim=1, keepdim=True)
    p2 = alpha(u[:, 3:]) * u2

    return 0.5 * (p1 + p2)


def alpha(rho):
    """
    Compute the alpha parameter based on rho.
    """
    alpha_max = 2.5e4
    alpha_min = 2.5e-4
    q = 0.1
    return alpha_max + (alpha_min - alpha_max) * rho * (1 + q) / (rho + q)


def calculate_loss_multioutput(
    model_list,
    iteration,
    diff_method="Numerical",
    title="default",
    problem="doublepipe",
):
    """
    Calculate loss for multi-output models based on PDE, BC, and IC residuals.
    """
    # Clone collocation points
    collocation_x = model_list[0].collocation_x.clone()

    # Compute mean values at collocation points
    m_col = model_list[0].mean_module_NN_All(collocation_x)

    # Evaluate covariance matrices if not already done
    if model_list[0].k_xX is None:
        for i, model in enumerate(model_list):
            model.k_xX = model.covar_module(
                model.train_inputs[0], collocation_x
            ).evaluate()

    # Perform Cholesky decomposition if not already done
    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-7):
            for model in model_list:
                model.chol_decomp = model.covar_module(model.train_inputs[0]).cholesky()

    # Solve for offsets using Cholesky decomposition
    offsets = []
    for i, model in enumerate(model_list):
        target_offset = model.train_targets.unsqueeze(-1) - model.mean_module_NN_All(
            model.train_inputs[0]
        )[:, i].unsqueeze(-1)
        offsets.append(model.chol_decomp._cholesky_solve(target_offset))

    # Compute values for velocity, pressure, and density
    ro_tensor = (
        m_col[:, 3].unsqueeze(-1) + model_list[3].k_xX.t() @ offsets[3]
    ).squeeze(-1)
    ro = modified_sigmoid(ro_tensor)

    u = (m_col[:, 0].unsqueeze(-1) + model_list[0].k_xX.t() @ offsets[0]).squeeze(-1)
    v = (m_col[:, 1].unsqueeze(-1) + model_list[1].k_xX.t() @ offsets[1]).squeeze(-1)
    p = (m_col[:, 2].unsqueeze(-1) + model_list[2].k_xX.t() @ offsets[2]).squeeze(-1)

    z_all = torch.cat(
        (u.unsqueeze(1), v.unsqueeze(1), p.unsqueeze(1), ro.unsqueeze(1)), dim=1
    )
    f = alpha(z_all[:, 3:]) * z_all[:, :2]
    fx, fy = f[:, :1], f[:, 1:]

    # Compute residuals
    if diff_method == "Numerical":
        dx, dy = 0.01, 0.01
        nx, ny = 100, 100
        u, v, p = [arr.reshape(nx, ny) for arr in (u, v, p)]
        fx, fy = [arr.reshape(nx, ny) for arr in (fx, fy)]
        residuals = compute_fvm_residuals_higher_order(
            u, v, p, fx, fy, nu=1, dx=dx, dy=dy
        )
        residual_pde1, residual_pde2, residual_pde3, u_x, v_y, u_y, v_x = residuals
        residual_pde1, residual_pde2, residual_pde3 = [
            res * w for res, w in zip(residuals[:3], (w_1, w_2, w_3))
        ]
        u, v, p = [arr.reshape(-1) for arr in (u, v, p)]
        fx, fy = [arr.reshape(-1) for arr in (fx, fy)]
    else:
        u_x, u_y, u_xx, u_yy = compute_autograd_derivatives(
            model_list[0], z_all[:, 0], collocation_x, grad_order=2
        )
        v_x, v_y, v_xx, v_yy = compute_autograd_derivatives(
            model_list[1], z_all[:, 1], collocation_x, grad_order=2
        )
        p_x, p_y, _, _ = compute_autograd_derivatives(
            model_list[2], z_all[:, 2], collocation_x, grad_order=1
        )
        residual_pde1 = (-(u_xx + u_yy) + p_x + fx[:, 0]) * w_1
        residual_pde2 = (-(v_xx + v_yy) + p_y + fy[:, 0]) * w_2
        residual_pde3 = (u_x + v_y) * w_3

    # Plot fields at checkpoints
    if iteration in checkpoints:
        plot_density_and_velocity_fields(u, v, p, ro, collocation_x, iteration, problem)
        plot_predictions_and_residuals(
            u,
            v,
            p,
            ro,
            collocation_x,
            iteration,
            residual_pde1,
            residual_pde2,
            residual_pde3,
            1,
            1,
            1,
            problem,
        )
    # Calculate losses
    loss_pde1 = torch.mean(residual_pde1**2)
    loss_pde2 = torch.mean(residual_pde2**2)
    loss_pde3 = torch.mean(residual_pde3**2)
    dp_loss = dissipated_power(
        z_all,
        u_x.reshape(-1, 1),
        v_y.reshape(-1, 1),
        u_y.reshape(-1, 1),
        v_x.reshape(-1, 1),
    )

    gamma = next((val for key, val in gamma_values.items() if key in title), 0.5)
    vol_loss = loss_volume(z_all[:, 3:], gamma=gamma) * w_4

    # Return final losses
    return loss_pde1, loss_pde2, loss_pde3, torch.sum(dp_loss), vol_loss


def find_TO(
    model_list,
    lr_default: float = 0.01,
    num_iter: int = 500,
    title: str = "default",
    problem: str = "doublepipe",  # 设置默认值
    diff_method: str = "Numerical",
) -> float:
    """
    Train models to minimize total loss using dynamic weights and record progress.

    Args:
        model_list: List of models to be trained.
        lr_default: Default learning rate for the optimizer.
        num_iter: Number of training iterations.
        title: Title for specific settings.
        diff_method: Differentiation method ('Numerical' or 'Autograd').

    Returns:
        loss history
    """
    # Use the first model as a reference for shared NN
    model_ref = model_list[0]
    for model in model_list:
        model.mean_module_NN_All = model_ref.mean_module_NN_All
        model.train()

    # Initialize variables
    loss_total, loss_hist, loss_hist_total = [], [], []
    loss_pde1_hist, loss_pde2_hist, loss_pde3_hist = [], [], []
    loss_dp_hist, vol_loss_hist = [], []
    scaled_loss_pde1_hist, scaled_loss_pde2_hist = [], []
    scaled_loss_pde3_hist, scaled_loss_dp_hist, scaled_vol_loss_hist = [], [], []
    weights = {"alpha": [], "beta": [], "mu_p": []}
    result_loss_thetas = {
        "largest_eigval_hist": [],
        "condition_number_hist": [],
        "Hessian": [],
        "grads": [],
    }
    GP_NN_hist, NN_hist, time_hist = [], [], []
    dynamic_weights = True
    sigma_1, beta_1, alph_1 = 10, 10, 0.1
    f_inc, mu_F = math.inf, 1

    # Initialize model-specific parameters
    model1 = model_list[0]
    model1.alpha, model1.beta, model1.theta = 1, 1, 1

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model_ref.parameters(), lr=lr_default)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=np.linspace(0, num_iter, 4).tolist(), gamma=0.75
    )

    # Training loop
    with tqdm(range(num_iter + 1), desc="Epoch", position=0, leave=True) as pbar:
        for j in pbar:
            optimizer.zero_grad()

            # Calculate losses
            loss_pde1, loss_pde2, loss_pde3, loss_dp, vol_loss = (
                calculate_loss_multioutput(
                    model_list, j, diff_method=diff_method, title=title, problem=problem
                )
            )
            loss_pde = loss_pde1 + loss_pde3

            # Dynamic weight adjustments
            if dynamic_weights:
                alpha, beta = compute_dynamic_weights_ic(
                    loss_pde1 + loss_pde2, loss_pde3, vol_loss, model1
                )
                if all(
                    torch.is_tensor(val)
                    and not (torch.isnan(val).any() or torch.isinf(val).any())
                    for val in [alpha, beta]
                ):
                    model1.alpha, model1.beta = alpha, beta

                weights["alpha"].append(alpha.detach().cpu().item())
                weights["beta"].append(beta.detach().cpu().item())
                weights["mu_p"].append(mu_F)

                loss = loss_dp + mu_F * (
                    loss_pde1
                    + loss_pde2
                    + model1.alpha * loss_pde3
                    + model1.beta * vol_loss
                )

                if (j + 1) % 50 == 0:
                    mu_F = min(mu_F * 1.05, 5e2)
            else:
                alph_1 = 1e-6 / (j + 1)
                beta_1, sigma_1 = 2 * (j + 1), 5 * (j + 1)
                loss = sigma_1 * loss_pde + alph_1 * loss_dp + beta_1 * vol_loss

            # Record loss history
            loss_total.append(loss.item())
            loss_pde1_hist.append((1 / w_1) * loss_pde1.item())
            loss_pde2_hist.append((1 / w_2) * loss_pde2.item())
            loss_pde3_hist.append((1 / w_3) * loss_pde3.item())
            loss_dp_hist.append((1 / w_5) * loss_dp.item())
            vol_loss_hist.append((1 / w_4) * vol_loss.item())
            scaled_loss_pde1_hist.append(loss_pde1.item())
            scaled_loss_pde2_hist.append(loss_pde2.item())
            scaled_loss_pde3_hist.append(model1.alpha * loss_pde3.item())
            scaled_loss_dp_hist.append(loss_dp.item())
            scaled_vol_loss_hist.append(model1.beta * vol_loss.item())

            # Plot loss history at checkpoints
            if j in checkpoints:
                plot_loss_history(
                    loss_total,
                    [loss_pde1_hist, scaled_loss_pde1_hist],
                    [loss_pde2_hist, scaled_loss_pde2_hist],
                    [loss_pde3_hist, scaled_loss_pde3_hist],
                    [loss_dp_hist, scaled_loss_dp_hist],
                    [vol_loss_hist, scaled_vol_loss_hist],
                    j,
                    problem,
                )

            # Backpropagation and optimizer step
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            # Update progress description
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            loss_hist.append(loss.item())

    loss_hist_total = loss_hist

    return loss_hist_total
