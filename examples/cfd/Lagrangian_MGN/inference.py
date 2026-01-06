import logging
import os
from functools import partial
from typing import Any

import hydra
import torch
import numpy as np
import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
matplotlib.use("Agg")

# OneScience imports
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd import graph_update
from onescience.datapipes.cfd import DeepMindLagrangianDatapipe
from onescience.launch.utils import load_checkpoint
from loggers import get_gpu_info, init_python_logging

logger = logging.getLogger("lmgn")

# 不同粒子类型的可视化颜色
TYPE_TO_COLOR = {
    0: "green",   # 刚体
    3: "black",   # 边界粒子
    5: "blue",    # 水
    6: "gold",    # 沙
    7: "magenta", # 粘性流体
}


class MGNRollout:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # 初始化分布式管理器（推理阶段通常为单卡，但保持接口一致）
        self.dist = DistributedManager()
        self.device = self.dist.device

        if cfg.test.batch_size != 1:
            raise ValueError(
                f"Only batch size 1 is currently supported, got {cfg.test.batch_size}"
            )

        self.frame_skip = cfg.inference.frame_skip

        # 初始化数据管道
        logger.info("Loading the test dataset...")
        self.datapipe = DeepMindLagrangianDatapipe(
            cfg, distributed=self.dist.distributed
        )
        self.dataloader = self.datapipe.test_dataloader()
        self.dataset = self.datapipe.test_dataset

        logger.info(f"Using {len(self.dataset)} test samples.")

        # 从数据集中读取元信息
        self.num_steps = self.dataset.num_steps
        self.dim = self.dataset.dim
        self.radius = self.dataset.radius
        self.dt = self.dataset.dt
        self.bounds = self.dataset.bounds
        self.num_history = self.dataset.num_history
        self.num_node_types = self.dataset.num_node_types

        # 初始化模型
        logger.info("Creating the model...")
        self.model = hydra.utils.instantiate(cfg.model)

        if cfg.compile.enabled:
            self.model = torch.compile(self.model, **cfg.compile.args)

        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载模型权重
        load_checkpoint(
            hydra.utils.to_absolute_path(cfg.resume_dir),
            models=self.model,
            device=self.device,
        )

    def compute_boundary_feature(self, position):
        """计算粒子到边界的距离特征"""
        dist = torch.cat(
            [position - self.bounds[0], self.bounds[1] - position], dim=-1
        )
        feat = torch.exp(-(dist**2) / self.radius**2)
        feat[dist > self.radius] = 0
        return feat

    def boundary_clamp(self, position):
        """对粒子位置进行边界裁剪"""
        min_bound = self.bounds[0] + 1e-3
        max_bound = self.bounds[1] - 1e-3
        return torch.clamp(position, min=min_bound, max=max_bound)

    def pack_inputs(self, position, vel_history, node_type):
        """重新组织模型输入特征"""
        bound_feat = self.compute_boundary_feature(position)
        vel_hist = vel_history.permute(1, 0, 2).flatten(1)
        return torch.cat((position, vel_hist, bound_feat, node_type), dim=-1)

    @torch.inference_mode()
    def predict(self):
        """对测试集进行完整时间序列 rollout"""
        pred_pos, gt_pos, node_type = [], [], []

        for graph in self.dataloader:
            graph = graph.to(self.device)

            # 新序列起始
            if graph.ndata["t"][0].item() == 0:
                if pred_pos:
                    yield torch.stack(pred_pos), torch.stack(gt_pos), node_type

                pred_pos, gt_pos, node_type = [], [], []

                position, vel_history, node_type = self.dataset.unpack_inputs(graph)
                position = position.clone()
                vel_history = vel_history.clone()

                pred_pos.append(position)
                gt_pos.append(position)

            # 使用当前预测状态更新图特征
            graph.ndata["x"] = self.pack_inputs(position, vel_history, node_type)
            graph.ndata["pos"] = position

            # 动态更新图连接关系
            graph_update(graph, self.radius)

            # 前向推理，预测加速度
            acceleration = self.model(
                graph.ndata["x"], graph.edata["x"], graph
            )

            # 时间积分
            next_pos, next_vel = self.dataset.time_integrator(
                position=position,
                velocity=vel_history[-1],
                acceleration=acceleration,
                dt=self.dt,
                denormalize=True,
            )

            next_pos = self.boundary_clamp(next_pos)

            # 速度重新归一化，用于历史队列
            next_vel_norm = (
                next_vel - self.dataset.vel_mean.to(self.device)
            ) / self.dataset.vel_std.to(self.device)

            vel_history = torch.cat(
                (vel_history[1:], next_vel_norm.unsqueeze(0)), dim=0
            )
            position = next_pos

            pred_pos.append(position)

            gt_next_pos, _, _ = self.dataset.unpack_targets(graph)
            gt_pos.append(gt_next_pos)

        if pred_pos:
            yield torch.stack(pred_pos), torch.stack(gt_pos), node_type


# ================= 可视化相关函数 =================

def init_animation(subplot_kw: dict[str, Any] = None):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 9), subplot_kw=subplot_kw
    )
    return fig, ax1, ax2


def plot_particles_2d(ax, title, position, node_color, bounds):
    ax.cla()
    ax.set_aspect("equal")
    ax.scatter(position[:, 0], position[:, 1], c=node_color, s=10)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_title(title)


def plot_particles_3d(ax, title, position, node_color, bounds):
    ax.cla()
    ax.scatter(position[:, 2], position[:, 0], position[:, 1],
               c=node_color, s=10)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_zlim(bounds[0], bounds[1])
    ax.set_title(title)


def animate(num, plotter, fig, ax1, ax2,
            pred, gt, node_color, bounds, frame_skip):
    num *= frame_skip
    if num >= len(pred):
        return
    plotter(ax1, "Prediction", pred[num], node_color, bounds)
    plotter(ax2, "Ground Truth", gt[num], node_color, bounds)
    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1
    )


def plot_error(mse, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_mse = np.mean(mse)
    ax.plot(mse, marker=".", label="Sequence MSE")
    ax.axhline(mean_mse, linestyle="--", color="red",
               label=f"Mean: {mean_mse:.4f}")
    ax.set_title("Lagrangian MeshGraphNet Rollout Error")
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Position MSE")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "error.png"))
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    init_python_logging(cfg, dist.rank)

    if dist.rank == 0:
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
        logger.info(get_gpu_info())
        logger.info("Rollout started...")

    rollout = MGNRollout(cfg)

    ani_dir = os.path.join(cfg.output, "animations")
    os.makedirs(ani_dir, exist_ok=True)

    mse_list = []

    for i, (pred_pos, gt_pos, node_type) in enumerate(rollout.predict()):
        logger.info(f"Processing sequence {i}...")

        pred = pred_pos.cpu().numpy()
        gt = gt_pos.cpu().numpy()
        node_type = node_type.cpu().numpy()

        node_color = [
            TYPE_TO_COLOR.get(idx, "gray")
            for idx in np.argmax(node_type, axis=1)
        ]

        error = np.mean((pred - gt) ** 2)
        mse_list.append(error)
        logger.info(f"Seq {i} MSE: {error:.4e}")

        if i < 5:
            if cfg.dim == 2:
                fig, ax1, ax2 = init_animation()
                plotter = plot_particles_2d
            elif cfg.dim == 3:
                fig, ax1, ax2 = init_animation(
                    subplot_kw={"projection": "3d"}
                )
                plotter = plot_particles_3d
            else:
                raise ValueError(f"Unsupported dim: {cfg.dim}")

            ani_func = partial(
                animate,
                plotter=plotter,
                fig=fig,
                ax1=ax1,
                ax2=ax2,
                pred=pred,
                gt=gt,
                node_color=node_color,
                bounds=rollout.bounds,
                frame_skip=rollout.frame_skip,
            )

            num_frames = (pred.shape[0] - 1) // rollout.frame_skip
            ani = animation.FuncAnimation(
                fig, ani_func, frames=num_frames, interval=100
            )
            ani.save(
                os.path.join(ani_dir, f"animation_{i}.gif"),
                writer="pillow",
                fps=10,
            )
            plt.close(fig)
            logger.info(f"Saved animation_{i}.gif")

    if dist.rank == 0:
        plot_error(mse_list, ani_dir)
        logger.info(f"Average MSE: {np.mean(mse_list):.4e}")
        logger.info("Inference completed!")

    dist.cleanup()


if __name__ == "__main__":
    main()
