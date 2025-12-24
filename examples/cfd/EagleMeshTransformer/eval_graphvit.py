import torch
import random
import os
import sys
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.animation as animation
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.datapipes import EagleDatapipe
from onescience.models.graphvit import GraphViT

def load_best_model(model, ckp_dir, device, model_name="best_model.pth"):
    """
    从指定目录加载最佳模型权重
    """
    ckpt_path = os.path.join(ckp_dir, model_name)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 兼容 DDP 和非 DDP 模式
        model_to_load = model.module if hasattr(model, "module") else model
        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        except KeyError:
            # 处理 checkpoint 仅包含状态字典的情况
            model_to_load.load_state_dict(checkpoint)
        logging.info(f"Successfully loaded model from {ckpt_path}")
    else:
        logging.error(f"Checkpoint file not found: {ckpt_path}. Exiting.")
        sys.exit(1)


def setup_logging(rank):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger

def create_animation(x, velocity, pressure, velocity_hat, pressure_hat, cells, idx, save_dir):
    v_magnitude = torch.sqrt((velocity**2).sum(-1)).squeeze(0).cpu().numpy()
    v_hat_magnitude = torch.sqrt((velocity_hat**2).sum(-1)).squeeze(0).cpu().numpy()
    mesh_pos = x["mesh_pos"].squeeze(0).cpu().numpy()
    triangles = cells.squeeze(0).cpu().numpy()[0] 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Simulation Comparison: Ground Truth vs Prediction (Sample {idx})")
    tri = Triangulation(mesh_pos[0, :, 0], mesh_pos[0, :, 1], triangles)
    vmin = min(v_magnitude.min(), v_hat_magnitude.min())
    vmax = max(v_magnitude.max(), v_hat_magnitude.max())
    plot1 = ax1.tripcolor(tri, v_magnitude[0], cmap="jet", vmin=vmin, vmax=vmax)
    ax1.set_title("Ground Truth Velocity Magnitude")
    ax1.set_axis_off()
    ax1.set_aspect("equal")
    fig.colorbar(plot1, ax=ax1)
    plot2 = ax2.tripcolor(tri, v_hat_magnitude[0], cmap="jet", vmin=vmin, vmax=vmax)
    ax2.set_title("Predicted Velocity Magnitude")
    ax2.set_axis_off()
    ax2.set_aspect("equal")
    fig.colorbar(plot2, ax=ax2)
    def update(frame):
        masked_tris = tri.get_masked_triangles()
        avg_magnitude1 = v_magnitude[frame][masked_tris].mean(axis=1)
        avg_magnitude2 = v_hat_magnitude[frame][masked_tris].mean(axis=1)
        plot1.set_array(avg_magnitude1)
        plot2.set_array(avg_magnitude2)
        fig.suptitle(f"Time Step: {frame}")
        return plot1, plot2
    anim = animation.FuncAnimation(
        fig, update, frames=len(v_magnitude), interval=50, blit=True
    )
    anim_path = save_dir / f"comparison_{idx}.gif"
    anim.save(anim_path, writer="pillow", fps=15)
    plt.close(fig)
    print(f"Saved animation to {anim_path}")


def main():
    # 初始化分布式环境、日志和计算设备
    DistributedManager.initialize()
    logger = setup_logging(0) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载配置文件
    config_file_path = "conf/graphvit_eagle.yaml"
    cfg_model = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")
    
    if hasattr(cfg_train, "max_anim_on_infer"):
        max_anim = cfg_train.max_anim_on_infer
    else:
        max_anim = 5 # 默认生成数量
        logger.warning(f"conf.training.max_anim_on_infer 未设置, 默认生成 {max_anim} 个动画。")

    logger.info(f"Loading config from: {config_file_path}")
    
    # 设置随机种子
    torch.manual_seed(3721)
    torch.cuda.manual_seed(3721)
    np.random.seed(3721)
    random.seed(3721)

    # 初始化数据管道
    logger.info("Initializing datapipe...")
    datapipe = EagleDatapipe(params=cfg_data, distributed=False)

    dataloader, _ = datapipe.test_dataloader(batch_size=1)
    
    dataset = dataloader.dataset
    L = dataset.w_len
    logger.info(f"Datapipe initialized. Test window length: {L}, N_cluster: {dataset.n_cluster}")

    # 初始化模型结构
    model = GraphViT(
        state_size=cfg_model.state_size, 
        w_size=cfg_model.w_size
    ).to(device)

    # 加载预训练权重
    checkpoint_dir = cfg_train.checkpoint_dir
    logger.info(f"Loading best model (best_model.pth) from dir: {checkpoint_dir}")
    
    load_best_model(
        model=model,
        ckp_dir=checkpoint_dir,
        device=device,
        model_name="best_model.pth"
    )
    
    animation_dir = Path("animation_results")
    animation_dir.mkdir(exist_ok=True)
    
    # 模型评估循环
    anim_count = 0
    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        model.eval()
        arange = torch.arange(1, L).to(device) 
        error_velocity = torch.zeros(L - 1).to(device)
        error_pressure = torch.zeros(L - 1).to(device)

        p_std = torch.sqrt((dataset.pressure_std**2).sum(-1)).to(device)
        v_std = torch.sqrt((dataset.velocity_std**2).sum(-1)).to(device)

        for i, x in enumerate(dataloader):
            if not x: 
                logger.warning(f"Skipping empty batch (index {i})")
                continue
                
            # 数据迁移至 GPU
            mesh_pos = x["mesh_pos"].to(device)
            edges = x["edges"].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()
            cells = x["cells"] 

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(
                mesh_pos,
                edges,
                state,
                node_type,
                clusters,
                clusters_mask,
                apply_noise=False,
            )

            # 反归一化处理
            velocity_hat, pressure_hat = state_hat[..., :2], state_hat[..., 2:]
            velocity_hat, pressure_hat = dataset.denormalize(velocity_hat, pressure_hat)
            velocity, pressure = dataset.denormalize(velocity, pressure)

            if anim_count < max_anim:
                create_animation(
                    x,
                    velocity.detach(),
                    pressure.detach(),
                    velocity_hat.detach(),
                    pressure_hat.detach(),
                    cells,
                    anim_count,
                    animation_dir 
                )
                anim_count += 1

            # 计算 N-RMSE
            v_gt, p_gt = velocity[0, 1:], pressure[0, 1:]
            v_hat, p_hat = velocity_hat[0, 1:], pressure_hat[0, 1:]
            mask = mask[0, 1:].unsqueeze(-1)
            rmse_velocity = torch.sqrt(((v_gt * mask - v_hat * mask) ** 2).sum(dim=-1))
            rmse_pressure = torch.sqrt(((p_gt * mask - p_hat * mask) ** 2).sum(dim=-1))
            rmse_velocity = rmse_velocity.mean(1)
            rmse_pressure = rmse_pressure.mean(1)
            rmse_velocity = torch.cumsum(rmse_velocity, dim=0) / arange
            rmse_pressure = torch.cumsum(rmse_pressure, dim=0) / arange
            error_velocity = error_velocity + rmse_velocity
            error_pressure = error_pressure + rmse_pressure
            error_v = error_velocity / (i + 1) / v_std
            error_p = error_pressure / (i + 1) / p_std
            error = error_p + error_v
            idx_50 = min(49, L-2)
            idx_last = L-2
            
            pbar.set_postfix(
                dict(
                    error_1=error[0].item(),
                    error_50=error[idx_50].item(),
                    error_last=error[idx_last].item(),
                )
            )
            pbar.update(1)
            
    logger.info(f"===== Final N-RMSE (T=1): {error[0].item():.6f} =====")
    logger.info(f"===== Final N-RMSE (T={idx_50+1}): {error[idx_50].item():.6f} =====")
    logger.info(f"===== Final N-RMSE (T={idx_last+1}): {error[idx_last].item():.6f} =====")
    logger.info(f"Animations saved to: {animation_dir.resolve()}")

if __name__ == "__main__":
    main()