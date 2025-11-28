# inference_meshgraphnet_dgl.py
#
# 适配 DGL 和 YParams 的 OneScience 推理脚本
# (已将 'Rollout' 重构为 'Inference')

import os
import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

# --- DGL 和绘图 ---
try:
    import dgl
    from dgl.dataloading import GraphDataLoader
except ImportError:
    raise ImportError("此脚本需要 DGL 库。")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import tri as mtri
from matplotlib.patches import Rectangle

# --- OneScience 核心 ---
from onescience.distributed.manager import DistributedManager 
from onescience.utils.YParams import YParams
from onescience.datapipes import DeepMind_CylinderFlowDatapipe
from onescience.launch.utils import load_checkpoint
from onescience.models.meshgraphnet import MeshGraphNet


def setup_logging(rank):
    """设置日志，只在 rank 0 输出 INFO"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger()


class MGNInference: # <--- [重命名]
    def __init__(
        self, 
        cfg_inference: YParams, # <--- [重命名]
        cfg_data: YParams, 
        cfg_train: YParams, 
        model_params: YParams, 
        logger: logging.Logger,
        dataloader: GraphDataLoader,
        dataset: "DeepMind_CylinderFlowDataset",
        stats: dict[str, any]
    ):
        
        # --- 1. 从 YParams 设置配置 ---
        self.num_test_time_steps = cfg_data.data.test_steps
        self.frame_skip = cfg_inference.frame_skip
        self.frame_interval = cfg_inference.frame_interval
        self.viz_vars = cfg_inference.viz_vars
        
        self.logger = logger

        # --- 2. 设置 Device ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using {self.device} device for inference") # <--- [重命名]

        # --- 3. 接收 Datapipe 组件 ---
        self.dataset = dataset
        self.dataloader = dataloader
        self.stats = {
            key: value.to(self.device) for key, value in stats['node_stats'].items()
        }

        # --- 4. 初始化模型 ---
        self.logger.info("Initializing model architecture...")
        mlp_act = "silu" if model_params.recompute_activation else "relu"
        
        self.model = MeshGraphNet(
            input_dim_nodes=model_params.num_input_features,
            input_dim_edges=model_params.num_edge_features,
            output_dim=model_params.num_output_features,
            mlp_activation_fn=mlp_act,
            do_concat_trick=model_params.do_concat_trick,
            num_processor_checkpoint_segments=model_params.num_processor_checkpoint_segments,
            recompute_activation=model_params.recompute_activation,
        )
        
        if cfg_train.jit:
            try:
                self.model = torch.jit.script(self.model).to(self.device)
            except Exception as e:
                self.logger.warning(f"JIT scripting failed, falling back: {e}")
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # --- 5. 加载 Checkpoint ---
        self.logger.info(f"Loading checkpoint from {cfg_train.checkpoint_dir}...")
        load_checkpoint(
            cfg_train.checkpoint_dir,
            models=self.model,
            device=self.device,
        )

        self.var_identifier = {"u": 0, "v": 1, "p": 2}

    def predict(self):
        self.pred, self.exact, self.faces, self.graphs = [], [], [], []
        
        self.logger.info("Starting auto-regressive inference...") # <--- [重命名]
        
        i = 0
        for data_tuple in self.dataloader:
            if not isinstance(data_tuple, (tuple, list)):
                # (假设的 Dataloader 逻辑)
                graph = data_tuple
                cells_idx = i // (self.num_test_time_steps - 1)
                if cells_idx >= len(self.dataset.cells):
                    self.logger.warning(f"Index {cells_idx} out of bounds for cells. Stopping.")
                    break
                cells = self.dataset.cells[cells_idx]
                mask = self.dataset.rollout_mask[cells_idx] # 'rollout_mask' 是原始数据字段名，保持不变
            else:
                graph, cells, mask = data_tuple

            graph = graph.to(self.device)
            
            # --- 规范化/反规范化 ---
            try:
                graph.ndata["x"][:, 0:2] = self.dataset.denormalize(
                    graph.ndata["x"][:, 0:2], self.stats["velocity_mean"], self.stats["velocity_std"]
                )
                graph.ndata["y"][:, 0:2] = self.dataset.denormalize(
                    graph.ndata["y"][:, 0:2],
                    self.stats["velocity_diff_mean"],
                    self.stats["velocity_diff_std"],
                )
                graph.ndata["y"][:, [2]] = self.dataset.denormalize(
                    graph.ndata["y"][:, [2]],
                    self.stats["pressure_mean"],
                    self.stats["pressure_std"],
                )
            except AttributeError as e:
                self.logger.error(f"Missing method on dataset class: {e}")
                raise e

            # --- 推理步骤 ---
            invar = graph.ndata["x"].clone()

            if i % (self.num_test_time_steps - 1) != 0:
                invar[:, 0:2] = self.pred[i - 1][:, 0:2].clone()
            
            try:
                invar[:, 0:2] = self.dataset.normalize_node(
                    invar[:, 0:2], self.stats["velocity_mean"], self.stats["velocity_std"]
                )
            except AttributeError as e:
                self.logger.error("Ensure DeepMind_CylinderFlowDataset has 'normalize_node' staticmethod.")
                raise e

            pred_i = self.model(invar, graph.edata["x"], graph).detach() 

            pred_i[:, 0:2] = self.dataset.denormalize(
                pred_i[:, 0:2], self.stats["velocity_diff_mean"], self.stats["velocity_diff_std"]
            )
            pred_i[:, 2] = self.dataset.denormalize(
                pred_i[:, 2], self.stats["pressure_mean"], self.stats["pressure_std"]
            )
            invar[:, 0:2] = self.dataset.denormalize(
                invar[:, 0:2], self.stats["velocity_mean"], self.stats["velocity_std"]
            )

            # Mask (原始数据字段 'rollout_mask' 保持不变)
            mask = torch.cat((mask, mask), dim=-1).to(self.device)
            pred_i[:, 0:2] = torch.where(
                mask, pred_i[:, 0:2], torch.zeros_like(pred_i[:, 0:2])
            )

            # 积分
            self.pred.append(
                torch.cat(
                    ((pred_i[:, 0:2] + invar[:, 0:2]), pred_i[:, [2]]), dim=-1
                ).cpu()
            )
            self.exact.append(
                torch.cat(
                    (
                        (graph.ndata["y"][:, 0:2] + graph.ndata["x"][:, 0:2]),
                        graph.ndata["y"][:, [2]],
                    ),
                    dim=-1,
                ).cpu()
            )

            self.faces.append(torch.squeeze(cells).numpy())
            self.graphs.append(graph.cpu())
            
            i += 1
            if i % 100 == 0 and self.logger.level == logging.INFO:
                # <--- [重命名] ---
                print(f"  Inference step {i}/{self.dataset.length}", end="\r")
        
        # <--- [重命名] ---
        self.logger.info(f"\nInference complete. Total steps: {i}") 

    def get_raw_data(self, idx):
        self.pred_i = [var[:, idx] for var in self.pred]
        self.exact_i = [var[:, idx] for var in self.exact]
        return self.graphs, self.faces, self.pred_i, self.exact_i

    # ( ... animation_init 和 animate 方法保持不变 ...)
    def init_animation(self, idx):
        self.pred_i = [var[:, idx] for var in self.pred]
        self.exact_i = [var[:, idx] for var in self.exact]

        plt.rcParams["image.cmap"] = "inferno"
        self.fig, self.ax = plt.subplots(2, 1, figsize=(16, 9))
        self.fig.set_facecolor("black")
        self.ax[0].set_facecolor("black")
        self.ax[1].set_facecolor("black")

        if not os.path.exists("./animations"):
            os.makedirs("./animations")

    def animate(self, num):
        num *= self.frame_skip
        graph = self.graphs[num]
        y_star = self.pred_i[num].numpy()
        y_exact = self.exact_i[num].numpy()
        
        triang = mtri.Triangulation(
            graph.ndata["mesh_pos"][:, 0].numpy(),
            graph.ndata["mesh_pos"][:, 1].numpy(),
            self.faces[num],
        )
        
        self.ax[0].cla()
        self.ax[0].set_aspect("equal")
        self.ax[0].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[0].add_patch(navy_box)
        self.ax[0].tripcolor(triang, y_star, vmin=np.min(y_star), vmax=np.max(y_star))
        self.ax[0].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[0].set_title("onescience MeshGraphNet Prediction", color="white")
        
        self.ax[1].cla()
        self.ax[1].set_aspect("equal")
        self.ax[1].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[1].add_patch(navy_box)
        self.ax[1].tripcolor(
            triang, y_exact, vmin=np.min(y_exact), vmax=np.max(y_exact)
        )
        self.ax[1].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[1].set_title("Ground Truth", color="white")

        self.ax[0].set_aspect("auto", adjustable="box")
        self.ax[1].set_aspect("auto", adjustable="box")
        self.ax[0].autoscale(enable=True, tight=True)
        self.ax[1].autoscale(enable=True, tight=True)
        self.fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.2
        )
        return self.fig


def main():
    # --- 1. 初始化 Manager 和 Logger ---
    DistributedManager.initialize() 
    manager = DistributedManager()
    logger = setup_logging(manager.rank)
    
    # --- 2. 加载 YParams 配置 ---
    config_file_path = "conf/mgn_cylinderflow.yaml"
    logger.info(f"Loading config from {config_file_path}")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_model = YParams(config_file_path, "model")
    cfg_train = YParams(config_file_path, "training")
    # <--- [重命名] ---
    cfg_inference = YParams(config_file_path, "inference") 
    
    model_params = cfg_model.specific_params[cfg_model.name]

    # --- 3. 初始化 Datapipe (DGL 版本) ---
    logger.info("Initializing datapipe (DGL)...")
    datapipe = DeepMind_CylinderFlowDatapipe(params=cfg_data, distributed=False)
    
    test_dataloader = datapipe.test_dataloader()
    test_dataset = datapipe.test_dataset
    stats = datapipe.stats
    
    logger.info("Datapipe initialized.")

    # --- 4. 初始化 Inference 类 ---
    # <--- [重命名] ---
    inference = MGNInference(
        cfg_inference, 
        cfg_data, 
        cfg_train, 
        model_params, 
        logger,
        test_dataloader,
        test_dataset,
        stats
    )
    
    # --- 5. 执行预测和可视化 ---
    # <--- [重命名] ---
    logger.info("Inference started...")
    idx = [inference.var_identifier[k] for k in inference.viz_vars]
    inference.predict()

    for i in idx:
        var_name = inference.viz_vars[i]
        logger.info(f"Creating animation for {var_name}...")
        inference.init_animation(i)
        
        ani = animation.FuncAnimation(
            inference.fig,
            inference.animate,
            frames=len(inference.graphs) // inference.frame_skip,
            interval=inference.frame_interval,
        )
        
        save_path = f"animations/animation_{var_name}.gif"
        ani.save(save_path)
        logger.info(f"Saved animation: {save_path}")

    logger.info("Inference finished.") # <--- [重命名]


if __name__ == "__main__":
    main()