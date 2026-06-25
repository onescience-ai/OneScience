"""
MeshGraphNet Distributed Model Builder

This module contains the dynamic model builder and GNN data structure for
MeshGraphNet distributed training with Megatron-LM.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

try:
    import dgl
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "MeshGraphNet requires the DGL library. Install it from: https://www.dgl.ai/"
    )
from dataclasses import dataclass
from onescience.distributed.megatron.core import mpu
from onescience.modules.meta import ModelMetaData
from onescience.modules.module import Module

from .meshgraphnet_stage0 import MeshGraphNetStage0
from .meshgraphnet_stage1 import MeshGraphNetStage1
from .meshgraphnet_stage2 import MeshGraphNetStage2


@dataclass
class MetaData(ModelMetaData):
    name: str = "MeshGraphNetDistributed"
    # Optimization, no JIT as DGLGraph causes trouble
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True

class MeshGraphNetDistributedStage(Module):
    """
    Dynamic MeshGraphNet Stage for Megatron-LM pipeline parallelism

    This class dynamically creates the appropriate stage based on pipeline rank.

    Args:
        config: Megatron config object
        input_dim_nodes: Input node feature dimension
        input_dim_edges: Input edge feature dimension
        output_dim: Output feature dimension
        processor_size: Number of message passing layers
        hidden_dim_processor: Hidden dimension for processor
        num_layers_node_processor: Number of layers in node processor MLP
        num_layers_edge_processor: Number of layers in edge processor MLP
        hidden_dim_node_encoder: Hidden dimension for node encoder
        num_layers_node_encoder: Number of layers in node encoder
        hidden_dim_edge_encoder: Hidden dimension for edge encoder
        num_layers_edge_encoder: Number of layers in edge encoder
        hidden_dim_node_decoder: Hidden dimension for node decoder
        num_layers_node_decoder: Number of layers in node decoder
        aggregation: Aggregation method
        mlp_activation_fn: Activation function for MLP
        do_concat_trick: Whether to use concat trick
        recompute_activation: Whether to recompute activation
    """

    meta = MetaData()

    def __init__(
        self,
        config,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        hidden_dim_processor: int = 128,
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        aggregation: str = "sum",
        mlp_activation_fn: str = "relu",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__(meta=MetaData())
        self.config = config

        # Graph information for each stage
        self.graph = None
        self.num_nodes = None
        self.num_edges = None
        self.hidden_dim = hidden_dim_processor
        self.share_embeddings_and_output_weights = False
        self.pre_process = None


        # Get pipeline parallel rank
        pipeline_model_parallel_size = mpu.get_pipeline_model_parallel_world_size()
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()

        # Determine which stage to create
        if pipeline_model_parallel_size == 1:
            # Single stage: Encoder + Processor + Decoder
            self.stage_type = "full"
            self.stage0 = MeshGraphNetStage0(
                input_dim_nodes=input_dim_nodes,
                input_dim_edges=input_dim_edges,
                hidden_dim_processor=hidden_dim_processor,
                hidden_dim_node_encoder=hidden_dim_node_encoder,
                num_layers_node_encoder=num_layers_node_encoder,
                hidden_dim_edge_encoder=hidden_dim_edge_encoder,
                num_layers_edge_encoder=num_layers_edge_encoder,
                mlp_activation_fn=mlp_activation_fn,
                recompute_activation=recompute_activation,
                config=config,
            )
            self.stage1 = MeshGraphNetStage1(
                processor_size=processor_size,
                input_dim_node=hidden_dim_processor,
                input_dim_edge=hidden_dim_processor,
                num_layers_node=num_layers_node_processor,
                num_layers_edge=num_layers_edge_processor,
                aggregation=aggregation,
                mlp_activation_fn=mlp_activation_fn,
                do_concat_trick=do_concat_trick,
                recompute_activation=recompute_activation,
                config=config,
            )
            self.stage2 = MeshGraphNetStage2(
                output_dim=output_dim,
                hidden_dim_processor=hidden_dim_processor,
                hidden_dim_node_decoder=hidden_dim_node_decoder,
                num_layers_node_decoder=num_layers_node_decoder,
                mlp_activation_fn=mlp_activation_fn,
                recompute_activation=recompute_activation,
                config=config,
            )
        elif pipeline_model_parallel_size == 2:
            # Two stages: Encoder (Stage 0) + Processor + Decoder (Stage 1)
            if pipeline_rank == 0:
                self.stage_type = "encoder"
                self.stage0 = MeshGraphNetStage0(
                    input_dim_nodes=input_dim_nodes,
                    input_dim_edges=input_dim_edges,
                    hidden_dim_processor=hidden_dim_processor,
                    hidden_dim_node_encoder=hidden_dim_node_encoder,
                    num_layers_node_encoder=num_layers_node_encoder,
                    hidden_dim_edge_encoder=hidden_dim_edge_encoder,
                    num_layers_edge_encoder=num_layers_edge_encoder,
                    mlp_activation_fn=mlp_activation_fn,
                    recompute_activation=recompute_activation,
                    config=config,
                )
            else:
                self.stage_type = "processor_decoder"
                self.stage1 = MeshGraphNetStage1(
                    processor_size=processor_size,
                    input_dim_node=hidden_dim_processor,
                    input_dim_edge=hidden_dim_processor,
                    num_layers_node=num_layers_node_processor,
                    num_layers_edge=num_layers_edge_processor,
                    aggregation=aggregation,
                    mlp_activation_fn=mlp_activation_fn,
                    do_concat_trick=do_concat_trick,
                    recompute_activation=recompute_activation,
                    config=config,
                )
                self.stage2 = MeshGraphNetStage2(
                    output_dim=output_dim,
                    hidden_dim_processor=hidden_dim_processor,
                    hidden_dim_node_decoder=hidden_dim_node_decoder,
                    num_layers_node_decoder=num_layers_node_decoder,
                    mlp_activation_fn=mlp_activation_fn,
                    recompute_activation=recompute_activation,
                    config=config,
                )
        elif pipeline_model_parallel_size >= 3:
            num_processor_stages = pipeline_model_parallel_size - 2
            layers_per_stage = processor_size // num_processor_stages
            remainder = processor_size % num_processor_stages
            
            if pipeline_rank == 0:
                # Stage 0: Encoder
                self.stage_type = "encoder"
                self.stage0 = MeshGraphNetStage0(
                    input_dim_nodes=input_dim_nodes,
                    input_dim_edges=input_dim_edges,
                    hidden_dim_processor=hidden_dim_processor,
                    hidden_dim_node_encoder=hidden_dim_node_encoder,
                    num_layers_node_encoder=num_layers_node_encoder,
                    hidden_dim_edge_encoder=hidden_dim_edge_encoder,
                    num_layers_edge_encoder=num_layers_edge_encoder,
                    mlp_activation_fn=mlp_activation_fn,
                    recompute_activation=recompute_activation,
                    config=config,
                )
            elif pipeline_rank == pipeline_model_parallel_size - 1:
                # Last stage: Decoder
                self.stage_type = "decoder"
                self.stage2 = MeshGraphNetStage2(
                    output_dim=output_dim,
                    hidden_dim_processor=hidden_dim_processor,
                    hidden_dim_node_decoder=hidden_dim_node_decoder,
                    num_layers_node_decoder=num_layers_node_decoder,
                    mlp_activation_fn=mlp_activation_fn,
                    recompute_activation=recompute_activation,
                    config=config,
                )
            else:
                self.stage_type = "processor"
                processor_stage_idx = pipeline_rank - 1  
                start_layer = processor_stage_idx * layers_per_stage
                start_layer += min(processor_stage_idx, remainder)
                
                end_layer = (processor_stage_idx + 1) * layers_per_stage
                end_layer += min(processor_stage_idx + 1, remainder)
                
                self.stage1 = MeshGraphNetStage1(
                    processor_size=processor_size,
                    layer_start=start_layer,
                    layer_end=end_layer,
                    input_dim_node=hidden_dim_processor,
                    input_dim_edge=hidden_dim_processor,
                    num_layers_node=num_layers_node_processor,
                    num_layers_edge=num_layers_edge_processor,
                    aggregation=aggregation,
                    mlp_activation_fn=mlp_activation_fn,
                    do_concat_trick=do_concat_trick,
                    recompute_activation=recompute_activation,
                    config=config,
                )
        else:
            raise ValueError(f"Invalid pipeline_model_parallel_size: {pipeline_model_parallel_size}")

    def set_graph_info(self, graph, num_nodes: int, num_edges: int):
        """
        Set graph information

        Each stage will call this method to store the graph object.
        This allows each stage to use the original graph object.

        Args:
            graph: DGLGraph object
            num_nodes: Number of nodes
            num_edges: Number of edges
        """
        self.graph = graph
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def _to_real_node(self, f): return f[:self.num_nodes].clone()
    def _to_real_edge(self, f): return f[:self.num_edges].clone()

    def _to_fixed_node(self, real_feat, max_len: int):
        fixed_shape = (max_len, real_feat.size(-1))
        out = torch.zeros(fixed_shape, device=real_feat.device, dtype=real_feat.dtype)
        out[:self.num_nodes] = real_feat
        return out
    
    def _to_fixed_edge(self, real_feat, max_len: int):
        fixed_shape = (max_len, real_feat.size(-1))
        out = torch.zeros(fixed_shape, device=real_feat.device, dtype=real_feat.dtype)
        out[:self.num_edges] = real_feat
        return out
    
    def forward(self, x):
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        is_first_stage = pp_rank == 0

        # print(f"[DEBUG] Rank {pp_rank} | stage_type: {self.stage_type} | is_first_stage: {is_first_stage}")

        # ==================== 处理输入 ====================
        if is_first_stage:
            node_features, edge_features = x
            # print(f"[DEBUG] Rank {pp_rank} | 解析成功: node={node_features.shape}, edge={edge_features.shape}")
        else:
            node_features, edge_features = self.input_tensor
            # print(f"[DEBUG] Rank {pp_rank} | 从流水线接收: node={node_features.shape}, edge={edge_features.shape}")

        if self.stage_type == "full":
            # print(f"[DEBUG] Rank {pp_rank} | full 输入: node={node_features.shape}, edge={edge_features.shape}")
            # -------- Stage 0: Encoder --------
            node_features, edge_features = self.stage0(node_features, edge_features)

            # print(f"[DEBUG] Rank {pp_rank} | stage0 encoder 输出: node={node_features.shape}, edge={edge_features.shape}")

            # -------- Stage 1: Processor --------
            # print(f"[DEBUG] Rank {pp_rank} | stage1 processor 输入: graph node={self.graph.num_nodes()}, edge={self.graph.num_edges()}")
            node_features, edge_features = self.stage1(node_features, edge_features, self.graph)
            # print(f"[DEBUG] Rank {pp_rank} | stage1 输出: node={node_features.shape}, edge={edge_features.shape}")

            # -------- Stage 2: Decoder --------

            output = self.stage2(node_features, edge_features)
            # print(f"[DEBUG] Rank {pp_rank} | stage2 输出: node={node_features.shape}, edge={edge_features.shape}")
            return output
        
        # ==================== Stage 0: Encoder ====================
        if self.stage_type == "encoder":
            # print(f"[DEBUG] Rank {pp_rank} | encoder 输入: node={node_features.shape}, edge={edge_features.shape}")

            node_real = self._to_real_node(node_features).contiguous()
            edge_real = self._to_real_edge(edge_features).contiguous()

            # print(f"Rank {pp_rank} [Stage0 forward] 切片后 node_real shape: {node_real.shape}, edge_real shape:{edge_real.shape}")
            node_real, edge_real = self.stage0(node_real, edge_real)

            node_features = self._to_fixed_node(node_real, node_features.size(0)).contiguous()
            edge_features = self._to_fixed_edge(edge_real, edge_features.size(0)).contiguous()

            # print(f"[DEBUG] Rank {pp_rank} | encoder 输出: node={node_features.shape}, edge={edge_features.shape}")
            return node_features, edge_features

        # ==================== Processor 系列阶段 ====================
        elif self.stage_type in ["processor", "processor_decoder"]:
            # print(f"Rank {pp_rank} [Stage1 forward] 输入 node_features shape: {node_features.shape}, edge_features shape: {edge_features.shape}")

            node_real = self._to_real_node(node_features).contiguous()
            edge_real = self._to_real_edge(edge_features).contiguous()

            # print(f"Rank {pp_rank} [Stage1 forward] 切片后 node_real shape: {node_real.shape}, edge_real shape:{edge_real.shape}")

            node_real, edge_real = self.stage1(node_real, edge_real, self.graph)

            node_features = self._to_fixed_node(node_real, node_features.size(0)).contiguous()
            edge_features = self._to_fixed_edge(edge_real, edge_features.size(0)).contiguous()
            
            # print(f"Rank {pp_rank} [Stage1 forward] 输出 node_features shape: {node_features.shape}, edge_features shape: {edge_features.shape}")

            if self.stage_type == "processor":
                # print(f"[DEBUG] Rank {pp_rank} | processor 输出: node={node_features.shape}, edge={edge_features.shape}")
                return node_features, edge_features

        # ==================== Decoder / 完整模型 ====================
        if self.stage_type in ["decoder", "processor_decoder"]:
            # print(f"Rank {pp_rank} [Stage2 forward] 输入 node_features shape: {node_features.shape}, edge_features shape: {edge_features.shape}")
            
            node_real = self._to_real_node(node_features).contiguous()
            edge_real = self._to_real_edge(edge_features).contiguous()
            # print(f"Rank {pp_rank} [Stage1 forward] 切片后 node_real shape: {node_real.shape}, edge_real shape: {edge_real.shape}")
            
            output = self.stage2(node_real, edge_real)
            # print(f"[DEBUG] Rank {pp_rank} | 最终输出 shape: {output.shape}")
            return output

        raise ValueError(f"未知阶段类型：{self.stage_type}")


def build_meshgraphnet_distributed_model(config, **kwargs):
    """
    Build MeshGraphNet distributed model

    Args:
        config: Megatron config object
        **kwargs: Model parameters

    Returns:
        MeshGraphNetDistributedStage model
    """
    return MeshGraphNetDistributedStage(config=config, **kwargs)