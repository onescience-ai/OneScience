"""
Megatron-LM Distributed Training Script for MeshGraphNet

This script implements distributed training for MeshGraphNet using Megatron-LM's
3D parallelism (DP + TP + PP).
"""

import argparse
import os
import sys
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Megatron imports
from onescience.distributed.megatron.training import pretrain, get_args
from onescience.distributed.megatron.core import mpu
from onescience.distributed.megatron.training.arguments import core_transformer_config_from_args
from onescience.distributed.megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from onescience.distributed.megatron.core.utils import get_attr_wrapped_model

# OneScience imports
from onescience.datapipes.cfd import DeepMind_CylinderFlowDatapipe
from onescience.models.meshgraphnet_distributed import build_meshgraphnet_distributed_model
from onescience.utils.YParams import YParams
from onescience.distributed.pipelinetensorshapeconfig import PipelineTensorShapeConfig
from onescience.distributed.megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

MAX_NODE = 4096
MAX_EDGE = 24576


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
    
    # Dataset configuration (prefix with 'dataset-' to avoid conflicts with Megatron)
    # group.add_argument('--dataset-train-samples', type=int, default=400,
    #                    help='Number of training samples')
    # group.add_argument('--dataset-train-steps', type=int, default=300,
    #                    help='Number of time steps per training sample')
    # group.add_argument('--dataset-val-samples', type=int, default=10,
    #                    help='Number of validation samples')
    # group.add_argument('--dataset-val-steps', type=int, default=300,
    #                    help='Number of time steps per validation sample')
    # group.add_argument('--dataset-test-samples', type=int, default=10,
    #                    help='Number of test samples')
    # group.add_argument('--dataset-test-steps', type=int, default=300,
    #                    help='Number of time steps per test sample')
    # group.add_argument('--dataset-noise-std', type=float, default=0.02,
    #                    help='Standard deviation of Gaussian noise for training data')
    group.add_argument('--lr-decay-rate', type=float, default=0.9999991,
                       help='Learning rate decay rate for MeshGraphNet (LambdaLR: decay_rate^step)')

    return parser


def model_provider(pre_process=False, post_process=True):
    """
    Build and provide the MeshGraphNet model for Megatron training

    Args:
        pre_process: Whether to include preprocessing (not used)
        post_process: Whether to include postprocessing (not used)

    Returns:
        MeshGraphNetDistributedStage model
    """
    args = get_args()
    config = core_transformer_config_from_args(args)

    model = build_meshgraphnet_distributed_model(
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

    # Configure pipeline tensor shapes for tuple communication
    # Based on Pangu Weather's approach: each stage returns a tuple of tensors
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    if pp_size > 1:
        single_stage = [
            [MAX_NODE, args.hidden_dim_processor],
            [MAX_EDGE, args.hidden_dim_processor],
        ]
        shapes = [single_stage for _ in range(pp_size - 1)]
        pp_config = PipelineTensorShapeConfig(num_stages=pp_size, stage_shapes=shapes)
    
    else:
        pp_config = None

    if pp_config is not None:
        args.pipeline_tensor_shape_config = pp_config

    return model


def compute_loss(output, targets, n_node=MAX_NODE):
    """
    Compute loss between model output and targets

    Args:
        output: Model predictions (num_nodes, output_dim)
        targets: Ground truth targets (num_nodes, output_dim)

    Returns:
        Tuple of (loss, num_tokens, metrics_dict)
        - loss: MSE loss value
        - num_tokens: Number of tokens (always 1 for GNN)
        - metrics_dict: Dictionary with 'lm loss' metric
    """
    loss_criterion = nn.MSELoss()
    loss = loss_criterion(output, targets)

    num_tokens = torch.tensor(n_node, dtype=torch.long, device=torch.cuda.current_device())
    reporting_loss = torch.stack([loss.detach() * num_tokens, num_tokens])
    # num_tokens = torch.tensor(1, device="cuda")
    # reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return loss, num_tokens, {'lm loss': reporting_loss}


def forward_step_func(data_iterator, model):
    graph = None
    node_feat = edge_feat = target = None
    n_node = n_edge = 0
    src = dst = None

    import dgl
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    is_pp_first = mpu.is_pipeline_first_stage()
    is_pp_last = mpu.is_pipeline_last_stage()
    device = torch.cuda.current_device()
   
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                src=mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )
    
    if tp_rank == 0:
        # 1. 读取图
        batch = next(data_iterator)
        graph = batch[0] if isinstance(batch, list) else batch
        graph = graph.to(device, non_blocking=True)
        
        # 2. 取出结构信息
        n_node = graph.num_nodes()
        n_edge = graph.num_edges()
        src, dst = graph.edges()
        src = src.to(device)
        dst = dst.to(device)    
        
        # 3. 取出特征 + label（必须在广播前拿出来！）
        node_feat = graph.ndata["x"].float().to(device)
        edge_feat = graph.edata["x"].float().to(device)
        target = graph.ndata["y"].float().to(device)
        
        # print(f"[DEBUG TP{tp_rank} PP{pp_rank}] 图解析完成 node={n_node} edge={n_edge}, src={src}, dst={dst}")
        # 4. 构造要广播的 tensor
        # 【顺序 1：数量】
        n_node_ts = torch.tensor([n_node], dtype=torch.long, device=device)
        n_edge_ts = torch.tensor([n_edge], dtype=torch.long, device=device)
        dim_n_ts = torch.tensor([node_feat.size(-1)], dtype=torch.long, device=device)
        dim_e_ts = torch.tensor([edge_feat.size(-1)], dtype=torch.long, device=device)
        dim_t_ts = torch.tensor([target.size(-1)], dtype=torch.long, device=device)
        _broadcast(n_node_ts)
        _broadcast(n_edge_ts)
        _broadcast(dim_n_ts)
        _broadcast(dim_e_ts)
        _broadcast(dim_t_ts)

        # 【顺序 2：结构】
        _broadcast(src)
        _broadcast(dst)

        # 【顺序 3：特征】
        _broadcast(node_feat)
        _broadcast(edge_feat)
        _broadcast(target)

        # print(f"[DEBUG TP{tp_rank} PP{pp_rank}] 广播完成 → node={n_node} edge={n_edge}, src={src}, dst={dst}")

    else:
        n_node_ts = torch.empty(1, dtype=torch.long, device=device)
        n_edge_ts = torch.empty(1, dtype=torch.long, device=device)
        dim_n_ts = torch.empty(1, dtype=torch.long, device=device)
        dim_e_ts = torch.empty(1, dtype=torch.long, device=device)
        dim_t_ts = torch.empty(1, dtype=torch.long, device=device)
        _broadcast(n_node_ts)
        _broadcast(n_edge_ts)
        _broadcast(dim_n_ts)
        _broadcast(dim_e_ts)
        _broadcast(dim_t_ts)

        n_node = n_node_ts.item()
        n_edge = n_edge_ts.item()
        dim_n = dim_n_ts.item()
        dim_e = dim_e_ts.item()
        dim_t = dim_t_ts.item()

        # -------------------- 第二步：用正确数量初始化 --------------------
        src = torch.empty(n_edge, dtype=torch.int32, device=device)
        dst = torch.empty(n_edge, dtype=torch.int32, device=device)
        node_feat = torch.empty((n_node, dim_n), dtype=torch.float32, device=device)
        edge_feat = torch.empty((n_edge, dim_e), dtype=torch.float32, device=device)
        target = torch.empty((n_node, dim_t), dtype=torch.float32, device=device)

        # -------------------- 第三步：再收 结构 --------------------
        _broadcast(src)
        _broadcast(dst)

        # -------------------- 第四步：最后收 特征 --------------------
        _broadcast(node_feat)
        _broadcast(edge_feat)
        _broadcast(target)

        # print(f"[DEBUG TP{tp_rank} PP{pp_rank}] 重建图 node={n_node} edge={n_edge}, src={src}, dst={dst}")
        graph = dgl.graph((src, dst), num_nodes=n_node).to(device)
        # print(f"[DEBUG TP{tp_rank} PP{pp_rank}] 图构造完成")

    # node_feat = graph.ndata["x"].float()
    # edge_feat = graph.edata["x"].float()
    # print(f"[DEBUG TP{tp_rank}] node_feat shape: {node_feat.shape}, edge_feat shape: {edge_feat.shape}")

    # ==========================
    # 4. 固定形状填充（仅 PP=0）
    # ==========================
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    if is_pp_first and pp_size > 1:
        # PP>1：必须填充到固定形状
        p_node = torch.zeros(MAX_NODE, node_feat.size(-1), device=device)
        p_edge = torch.zeros(MAX_EDGE, edge_feat.size(-1), device=device)
        p_node[:n_node] = node_feat
        p_edge[:n_edge] = edge_feat
        node_feat, edge_feat = p_node, p_edge

    # ==========================
    # 5. 模型前向
    # ==========================
    # print(f"[DEBUG TP{tp_rank}] 准备进入 model.forward")
    set_graph_info = get_attr_wrapped_model(model, "set_graph_info")
    set_graph_info(graph, n_node, n_edge)
    # print(f"[DEBUG TP{tp_rank}] node_feat shape: {node_feat.shape}, edge_feat shape: {edge_feat.shape}")
    output = model((node_feat, edge_feat))
    # print(f"[DEBUG TP{tp_rank}] model.forward 完成")

    loss_func = partial(compute_loss, targets=target, n_node=n_node) if is_pp_last else None
    # print(f"[DEBUG TP{tp_rank}] ====== forward_step_func 结束 ======")
    return output, loss_func

def train_valid_test_dataset_provider(train_valid_test_num_samples):
    """
    Provide train, validation, and test datasets

    Args:
        train_valid_test_num_samples: Tuple of (train_samples, val_samples, test_samples)

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
        Each dataloader is an iterator (not a DataLoader object)
    """
    args = get_args()

    config_file_path = "conf/mgn_cylinderflow.yaml"
    cfg_data = YParams(config_file_path, "datapipe")

    # Create datapipe
    datapipe = DeepMind_CylinderFlowDatapipe(params=cfg_data, distributed=True)

    # Get dataloaders
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()
    test_dataloader = datapipe.test_dataloader()

    train_valid_test_dataset_provider.train_sampler = train_sampler
    train_valid_test_dataset_provider.val_sampler = val_sampler

    stats = datapipe.stats
    args.steps_per_epoch = len(train_dataloader.dataset) // args.micro_batch_size
    print(f"[DEBUG] steps_per_epoch={args.steps_per_epoch}")

    # Return iterators (not DataLoaders) for Megatron's external dataloader mode
    return iter(train_dataloader), iter(val_dataloader), iter(test_dataloader)


def para_init():
    """
    Initialize parallel environment and set random seeds

    Returns:
        config: Megatron config object
    """
    import torch.distributed as dist
    import random
    import numpy as np

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    seed = 2222
    args = get_args()
    config = core_transformer_config_from_args(args)

    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return config


def main():
    """
    Main training function
    """
    # Set is_distributed attribute for dataset provider
    train_valid_test_dataset_provider.is_distributed = True

    original_get_lr = OptimizerParamScheduler.get_lr

    def custom_get_lr(self, param_group: dict) -> float:
        args = get_args()

        if hasattr(args, 'lr_decay_rate') and args.lr_decay_rate < 1.0:
            max_lr = param_group.get('max_lr', self.max_lr)
            min_lr = param_group.get('min_lr', self.min_lr)

            if not hasattr(args, 'steps_per_epoch'):
                return max_lr

            # current_epoch = self.num_steps / args.steps_per_epoch
            # lr = max_lr * (args.lr_decay_rate ** current_epoch)

            lr = max_lr * (args.lr_decay_rate ** self.num_steps)
            
            lr = max(min_lr, lr)
            return lr

        return original_get_lr(self, param_group)

    OptimizerParamScheduler.get_lr = custom_get_lr

    # Start training
    pretrain(
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
        model_provider=model_provider,
        model_type=None,
        forward_step_func=forward_step_func,
        extra_args_provider=add_meshgraphnet_args,
        args_defaults={'dataloader_type': 'external'},
    )


if __name__ == "__main__":
    main()