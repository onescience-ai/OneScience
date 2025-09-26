import os
import time

import numpy as np
import torch
from torch.optim import LBFGS, Adam

from onescience.models.mlp import FullyConnected

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 参数设置
E = 5.0
nu = 0.3
lmbd = (E * nu) / ((1 + nu) * (1 - nu))
mu = (E * 0.5) / (1 + nu)


def generate_data():
    # 域内点（矩形减去圆孔）
    n_domain = 40000
    coords = np.random.rand(n_domain, 2)
    radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    coords = coords[radius >= 0.2]

    # 边界点生成（确保不在孔洞内）
    n_boundary = 200

    # 右边界 (x=1)
    right = np.hstack(
        [np.ones((n_boundary, 1)), np.random.rand(n_boundary, 1)])

    # 左边界 (x=0, y >= 0.2)
    left = []
    while len(left) < n_boundary:
        temp = np.hstack(
            [np.zeros((n_boundary, 1)), np.random.rand(n_boundary, 1)])
        mask = temp[:, 1] >= 0.2
        left.extend(temp[mask])
    left = np.array(left[:n_boundary])
    # 顶部 (y=1, x >= 0.2)
    top = []
    while len(top) < n_boundary:
        temp = np.hstack(
            [np.random.rand(n_boundary, 1), np.ones((n_boundary, 1))])
        mask = temp[:, 0] >= 0.2
        top.extend(temp[mask])
    top = np.array(top[:n_boundary])

    # 底部 (y=0, x >= 0.2)
    bottom = []
    while len(bottom) < n_boundary:
        temp = np.hstack(
            [np.random.rand(n_boundary, 1), np.zeros((n_boundary, 1))])
        mask = temp[:, 0] >= 0.2
        bottom.extend(temp[mask])
    bottom = np.array(bottom[:n_boundary])
    # 圆孔边界
    theta = np.linspace(0, np.pi / 2, n_boundary)
    hole = 0.2 * \
        np.column_stack([np.cos(theta), np.sin(theta)])

    coords = np.concatenate(
        [coords, right, left, top, bottom, hole])
    return torch.tensor(coords, dtype=torch.float32).to(device)


model = FullyConnected(
    in_features=2, layer_size=64, out_features=5, num_layers=6, activation_fn="tanh"
).to(device)


def compute_loss(coords):
    coords.requires_grad = True
    u_pred = model(coords)

    ux = u_pred[:, 0]
    uy = u_pred[:, 1]
    sx = u_pred[:, 2]
    sy = u_pred[:, 3]
    sxy = u_pred[:, 4]

    # 应变计算
    E_xx = torch.autograd.grad(
        ux, coords, grad_outputs=torch.ones_like(ux), create_graph=True
    )[0][:, 0]
    E_yy = torch.autograd.grad(
        uy, coords, grad_outputs=torch.ones_like(uy), create_graph=True
    )[0][:, 1]
    E_xy = 0.5 * (
        torch.autograd.grad(
            ux, coords, grad_outputs=torch.ones_like(ux), create_graph=True
        )[0][:, 1]
        + torch.autograd.grad(
            uy, coords, grad_outputs=torch.ones_like(uy), create_graph=True
        )[0][:, 0]
    )

    # 本构关系
    S_xx = (2 * mu + lmbd) * E_xx + lmbd * E_yy
    S_yy = (2 * mu + lmbd) * E_yy + lmbd * E_xx
    S_xy = 2 * mu * E_xy

    # 平衡方程残差
    Sxx_x = torch.autograd.grad(
        sx, coords, grad_outputs=torch.ones_like(sx), create_graph=True
    )[0][:, 0]
    Sxy_y = torch.autograd.grad(
        sxy, coords, grad_outputs=torch.ones_like(sxy), create_graph=True
    )[0][:, 1]
    momentum_x = Sxx_x + Sxy_y

    Syy_y = torch.autograd.grad(
        sy, coords, grad_outputs=torch.ones_like(sy), create_graph=True
    )[0][:, 1]
    Sxy_x = torch.autograd.grad(
        sxy, coords, grad_outputs=torch.ones_like(sxy), create_graph=True
    )[0][:, 0]
    momentum_y = Sxy_x + Syy_y

    # 应力残差
    stress_x_res = sx - S_xx
    stress_y_res = sy - S_yy
    stress_xy_res = sxy - S_xy

    # PDE损失（域内点）
    mask_domain = (torch.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) > 0.2) & (
        (coords[:, 0] > 0) & (coords[:, 1] > 0)
    )  # 简单域内判断
    Equilibrium_loss = torch.mean(momentum_x[mask_domain] ** 2) + torch.mean(
        momentum_y[mask_domain] ** 2
    )
    Stress_loss = (
        torch.mean(stress_x_res[mask_domain] ** 2)
        + torch.mean(stress_y_res[mask_domain] ** 2)
        + torch.mean(stress_xy_res[mask_domain] ** 2)
    )
    # 边界条件损失
    # 右边界：σ_xx = sin(π/2 y), σ_xy=0
    mask_right = coords[:, 0] >= 0.999
    loss_sxx_right, loss_sxy_right = 0.0, 0.0
    if mask_right.sum() > 0:
        target_sxx = torch.sin(
            torch.pi / 2 * coords[mask_right, 1])
        loss_sxx_right = torch.mean(
            (sx[mask_right] - target_sxx) ** 2)
        loss_sxy_right = torch.mean(sxy[mask_right] ** 2)
        right_loss = loss_sxx_right + loss_sxy_right
    # 顶部：σ_yy=0，σ_xy=0
    mask_top = coords[:, 1] >= 0.999
    loss_syy_top, loss_sxy_top = 0.0, 0.0
    if mask_top.sum() > 0:
        loss_syy_top = torch.mean(sy[mask_top] ** 2)
        loss_sxy_top = torch.mean(sxy[mask_top] ** 2)
        top_loss = loss_syy_top + loss_sxy_top
    # 左边界：σ_xy=0
    mask_left = coords[:, 0] <= 0.001
    loss_sxy_left = 0.0
    if mask_left.sum() > 0:
        loss_sxy_left = torch.mean(sxy[mask_left] ** 2)
        loss_ux_left = torch.mean(ux[mask_left] ** 2)
        left_loss = loss_sxy_left + loss_ux_left
    # 底部：σ_xy=0
    mask_bottom = coords[:, 1] <= 0.001
    loss_sxy_bottom = 0.0
    if mask_bottom.sum() > 0:
        loss_sxy_bottom = torch.mean(sxy[mask_bottom] ** 2)
        loss_uy_bottom = torch.mean(uy[mask_bottom] ** 2)
        bottom_loss = loss_sxy_bottom + loss_uy_bottom
    # 圆孔边界：面力为零
    hole_radius = 0.2
    hole_mask = (
        torch.sqrt(coords[:, 0] ** 2 +
                   coords[:, 1] ** 2) - hole_radius
    ).abs() < 1e-4
    loss_traction_x, loss_traction_y = 0.0, 0.0
    if hole_mask.sum() > 0:
        # 法向量计算
        n_x = coords[hole_mask, 0] / hole_radius
        n_y = coords[hole_mask, 1] / hole_radius

        # 计算孔边界应力
        sx_hole = sx[hole_mask]
        sy_hole = sy[hole_mask]
        sxy_hole = sxy[hole_mask]
        t_x = sx_hole * n_x + sxy_hole * n_y
        t_y = sxy_hole * n_x + sy_hole * n_y

        traction_loss = torch.mean(t_x**2 + t_y**2)

    return (
        Equilibrium_loss,
        Stress_loss,
        right_loss,
        top_loss,
        left_loss,
        bottom_loss,
        traction_loss,
    )


def train():
    # 生成训练数据
    coords = generate_data()

    # Adam优化阶段
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.5)
    start_time = time.time()
    n_epochs = 5000
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        (
            Equilibrium_loss,
            Stress_loss,
            right_loss,
            top_loss,
            left_loss,
            bottom_loss,
            traction_loss,
        ) = compute_loss(coords)
        total_loss = (
            10 * Equilibrium_loss
            + Stress_loss
            + right_loss
            + top_loss
            + left_loss
            + bottom_loss
            + traction_loss
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}, Total Loss: {total_loss.item():.6f}, Equilibrium Loss : {Equilibrium_loss.item():.6f}, Stress Loss : {Stress_loss.item():.6f}, "
                f"Right Loss : {right_loss.item():.6f}, Left Loss : {left_loss.item():.6f}, Top Loss : {top_loss.item():.6f}, Bottom Loss : {bottom_loss.item():.6f}, "
                f'Hole Loss: {traction_loss.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    # L-BFGS优化阶段
    optimizer = LBFGS(model.parameters(
    ), max_iter=25000, line_search_fn="strong_wolfe")
    last_total_loss = [None]
    last_loss_terms = [None]  # 存储各项损失的容器

    def closure():
        optimizer.zero_grad()
        # 获取各项损失
        (
            Equilibrium_loss,
            Stress_loss,
            right_loss,
            top_loss,
            left_loss,
            bottom_loss,
            traction_loss,
        ) = compute_loss(coords)
        total_loss = (
            10 * Equilibrium_loss
            + Stress_loss
            + right_loss
            + top_loss
            + left_loss
            + bottom_loss
            + traction_loss
        )

        # 保存损失项
        last_loss_terms[0] = (
            Equilibrium_loss.item(),
            Stress_loss.item(),
            right_loss.item(),
            top_loss.item(),
            left_loss.item(),
            bottom_loss.item(),
            traction_loss.item(),
        )
        last_total_loss[0] = total_loss.item()

        total_loss.backward()
        return total_loss

    start_time = time.time()
    for stage in range(50):
        optimizer.step(closure)
        if last_loss_terms[0] is not None:
            eq, stress, right, top, left, bottom, hole = last_loss_terms[0]
            print(
                f"L-BFGS Stage [{stage+1}/50] Step {(stage+1)*500} | "
                f"Total Loss: {last_total_loss[0]:.6e} | "
                f"Equilibrium: {eq:.6e} | Stress: {stress:.6e} | "
                f"Right: {right:.6e} | Top: {top:.6e} | "
                f"Left: {left:.6e} | Bottom: {bottom:.6e} | Hole: {hole:.6e} | "
                f"Time: {time.time()-start_time:.1f}s"
            )
    # 训练完成后保存模型参数
    save_dir = "./model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, "model_state_dict.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model parameters saved to {save_path}")


# 执行训练
train()
