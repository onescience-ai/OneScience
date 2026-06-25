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

from onescience.distributed.manager import DistributedManager 
from onescience.utils.YParams import YParams
from onescience.datapipes.cfd import DeepMind_CylinderFlowDatapipe
# from onescience.launch.utils import load_checkpoint
from onescience.distributed.megatron.training import get_args
from onescience.distributed.megatron.training.arguments import core_transformer_config_from_args
from onescience.models.meshgraphnet import MeshGraphNet
from onescience.models.meshgraphnet_distributed import build_meshgraphnet_distributed_model

from onescience.distributed.megatron.core import parallel_state
from onescience.distributed.megatron.training.initialize import initialize_megatron
from onescience.distributed.megatron.training.checkpointing import load_checkpoint
from onescience.distributed.megatron.training.arguments import parse_args


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


class MGNInference: 
    def __init__(
        self,
        cfg_inference: YParams,
        cfg_data: YParams, 
        cfg_train: YParams, 
        model_params: YParams, 
        logger: logging.Logger,
        dataloader: GraphDataLoader,
        dataset: "DeepMind_CylinderFlowDataset",
        stats: dict[str, any]
    ):

        # --- 从 YParams 设置配置 ---
        self.num_test_time_steps = cfg_data.data.test_steps
        self.frame_skip = cfg_inference.frame_skip
        self.frame_interval = cfg_inference.frame_interval
        self.viz_vars = cfg_inference.viz_vars
        
        self.logger = logger

        # --- 设置 Device ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using {self.device} device for inference") 

        # --- 接收 Datapipe 组件 ---
        self.dataset = dataset
        self.dataloader = dataloader
        self.stats = {
            key: value.to(self.device) for key, value in stats['node_stats'].items()
        }

        # --- 初始化模型 ---
        self.logger.info("Initializing model architecture...")
        mlp_act = "silu" if model_params.recompute_activation else "relu"
        
        args = get_args()
        config = core_transformer_config_from_args(args)

        self.model = build_meshgraphnet_distributed_model(
            config=config,
            input_dim_nodes=args.input_dim_nodes,
            input_dim_edges=args.input_dim_edges,
            output_dim=args.output_dim,
            processor_size=args.processor_size,
            hidden_dim_processor=args.hidden_dim_processor,
            num_layers_node_processor=args.num_layers_node_processor,
            num_layers_edge_processor=args.num_layers_edge_processor,
            hidden_dim_node_encoder=args.hidden_dim_node_encoder,
            num_layers_node_encoder=args.num_layers_node_encoder,
            hidden_dim_edge_encoder=args.hidden_dim_edge_encoder,
            num_layers_edge_encoder=args.num_layers_edge_encoder,
            hidden_dim_node_decoder=args.hidden_dim_node_decoder,
            num_layers_node_decoder=args.num_layers_node_decoder,
            aggregation=args.aggregation,
            mlp_activation_fn=args.mlp_activation_fn,
            do_concat_trick=args.do_concat_trick,
            recompute_activation=args.recompute_activation,
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

        # --- 加载 Checkpoint ---
        self.logger.info(f"Loading checkpoint from {cfg_train.checkpoint_dir}...")
        # load_checkpoint(
        #     cfg_train.checkpoint_dir,
        #     models=self.model,
        #     device=self.device,
        # )
        load_checkpoint(self.model, None, None)

        self.var_identifier = {"u": 0, "v": 1, "p": 2}

    def predict(self):
        self.pred, self.exact, self.faces, self.graphs = [], [], [], []
        
        self.logger.info("Starting auto-regressive inference...") 
        
        i = 0
        for data_tuple in self.dataloader:
            if not isinstance(data_tuple, (tuple, list)):
                graph = data_tuple
                cells_idx = i // (self.num_test_time_steps - 1)
                if cells_idx >= len(self.dataset.cells):
                    self.logger.warning(f"Index {cells_idx} out of bounds for cells. Stopping.")
                    break
                cells = self.dataset.cells[cells_idx]
                mask = self.dataset.rollout_mask[cells_idx] 
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
                print(f"  Inference step {i}/{self.dataset.length}", end="\r")

        self.logger.info(f"\nInference complete. Total steps: {i}") 

    def get_raw_data(self, idx):
        self.pred_i = [var[:, idx] for var in self.pred]
        self.exact_i = [var[:, idx] for var in self.exact]
        return self.graphs, self.faces, self.pred_i, self.exact_i
        
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


def add_meshgraphnet_args(parser):
    """
    Add MeshGraphNet-specific arguments to the parser

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Modified parser
    """
    group = parser.add_argument_group(title='MeshGraphNet model arguments')

    group.add_argument('--input-dim-nodes', type=int, default=6,
                       help='Input dimension for node features')
    group.add_argument('--input-dim-edges', type=int, default=3,
                       help='Input dimension for edge features')
    group.add_argument('--output-dim', type=int, default=3,
                       help='Output dimension')
    group.add_argument('--processor-size', type=int, default=15,
                       help='Number of processor layers')
    group.add_argument('--hidden-dim-processor', type=int, default=128,
                       help='Hidden dimension for processor layers')
    group.add_argument('--num-layers-node-processor', type=int, default=2,
                       help='Number of layers in node processor MLP')
    group.add_argument('--num-layers-edge-processor', type=int, default=2,
                       help='Number of layers in edge processor MLP')
    group.add_argument('--hidden-dim-node-encoder', type=int, default=128,
                       help='Hidden dimension for node encoder')
    group.add_argument('--num-layers-node-encoder', type=int, default=2,
                       help='Number of layers in node encoder MLP')
    group.add_argument('--hidden-dim-edge-encoder', type=int, default=128,
                       help='Hidden dimension for edge encoder')
    group.add_argument('--num-layers-edge-encoder', type=int, default=2,
                       help='Number of layers in edge encoder MLP')
    group.add_argument('--hidden-dim-node-decoder', type=int, default=128,
                       help='Hidden dimension for node decoder')
    group.add_argument('--num-layers-node-decoder', type=int, default=2,
                       help='Number of layers in node decoder MLP')
    group.add_argument('--aggregation', type=str, default='sum',
                       choices=['sum', 'mean', 'max'],
                       help='Aggregation method for message passing')
    group.add_argument('--do-concat-trick', action='store_true', default=False,
                       help='Whether to use concatenation trick')
    group.add_argument('--num-processor-checkpoint-segments', type=int, default=0,
                       help='Number of checkpoint segments for processor layers')
    group.add_argument('--recompute-activation', action='store_true', default=False,
                       help='Whether to recompute activations')
    group.add_argument('--mlp-activation-fn', type=str, default='relu',
                       choices=['relu', 'silu', 'gelu'],
                       help='Activation function for MLP layers')
    group.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing training data')
    group.add_argument('--stats-dir', type=str, default=None,
                       help='Directory containing statistics for normalization')
    
    group.add_argument('--lr-decay-rate', type=float, default=0.9999991,
                       help='Learning rate decay rate for MeshGraphNet (LambdaLR: decay_rate^step)')

    return parser



def main():
    # --- 初始化 Manager 和 Logger ---
    # DistributedManager.initialize() 
    # manager = DistributedManager()
    # logger = setup_logging(manager.rank)

    initialize_megatron(extra_args_provider=add_meshgraphnet_args, args_defaults={
        # 必传，解决你所有报错
        'num_layers': 14,
        'hidden_size': 128,
        'num_attention_heads': 4,
        'max_position_embeddings': 1024,
        'seq_length': 1024,
        'tokenizer_type': 'NullTokenizer',
        'vocab_size': 1000,

        # 并行配置（从 train_slurm.sh 来）
        'tensor_model_parallel_size': 2,
        'pipeline_model_parallel_size': 1,
        'data_parallel_size': 1,

        # 批次
        'micro_batch_size': 1,
        'global_batch_size': 8,

        # 检查点
        'load': './checkpoints',
        'ckpt_format': 'torch_dist',
        'use_dist_ckpt': True,
        'checkpoint_dir': './checkpoints',

        # 指定加载 iter
        # 'load_iteration': 3000,
    })
    args = parse_args()
    rank = parallel_state.get_data_parallel_rank()

    logger = setup_logging(rank)

    # --- 加载 YParams 配置 ---
    config_file_path = "conf/mgn_cylinderflow.yaml"
    logger.info(f"Loading config from {config_file_path}")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_model = YParams(config_file_path, "model")
    cfg_train = YParams(config_file_path, "training")
    cfg_inference = YParams(config_file_path, "inference") 
    
    model_params = cfg_model.specific_params[cfg_model.name]

    # --- 初始化 Datapipe (DGL 版本) ---
    logger.info("Initializing datapipe (DGL)...")
    datapipe = DeepMind_CylinderFlowDatapipe(params=cfg_data, distributed=False)
    
    test_dataloader = datapipe.test_dataloader()
    test_dataset = datapipe.test_dataset
    stats = datapipe.stats
    
    logger.info("Datapipe initialized.")

    # --- 初始化 Inference 类 ---
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
    
    # --- 执行预测和可视化 ---
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

    logger.info("Inference finished.") 


if __name__ == "__main__":
    main()