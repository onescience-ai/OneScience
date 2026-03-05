import torch
import torch.nn as nn
from torch import Tensor

try:
    import dgl  # noqa: F401 for docs
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "Mesh Graph Net requires the DGL library. Install the "
    )
from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Tuple, Union

import onescience  # noqa: F401 for docs
from onescience.modules import OneEdge, OneNode, OneMlp

from onescience.modules.utils.gnnlayer_utils import CuGraphCSC, set_checkpoint_fn
from onescience.modules.layer.activations import get_activation
from onescience.modules.meta import ModelMetaData
from onescience.modules.module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "MeshGraphNet"
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


class MeshGraphNet(Module):
    """
        MeshGraphNet 网络架构。

        该模型基于 "Learning mesh-based simulation with graph networks" (Pfaff et al., 2020) 实现。
        它采用 Encode-Process-Decode 架构：
        1. **Encoder**: 将节点和边的物理特征映射到高维隐空间。
        2. **Processor**: 通过多层消息传递（Message Passing）在图中传播信息，更新节点和边的隐状态。
        3. **Decoder**: 将处理后的节点特征解码回物理空间（例如加速度或速度增量）。

        本实现完全基于 OneXxx 模块工厂构建，支持灵活的组件替换。

        Args:
            input_dim_nodes (int): 输入节点特征的维度。
            input_dim_edges (int): 输入边特征的维度。
            output_dim (int): 输出特征的维度（通常是节点状态的更新量）。
            processor_size (int, optional): 消息传递块（Processor Block）的数量。默认值: 15。
            mlp_activation_fn (Union[str, List[str]], optional): MLP 中使用的激活函数。默认值: 'relu'。
            num_layers_node_processor (int, optional): 处理器中节点更新 MLP 的层数。默认值: 2。
            num_layers_edge_processor (int, optional): 处理器中边更新 MLP 的层数。默认值: 2。
            hidden_dim_processor (int, optional): 处理器中隐层的特征维度。默认值: 128。
            hidden_dim_node_encoder (int, optional): 节点编码器的隐层维度。默认值: 128。
            num_layers_node_encoder (Union[int, None], optional): 节点编码器的层数。如果为 None，则不使用编码器。默认值: 2。
            hidden_dim_edge_encoder (int, optional): 边编码器的隐层维度。默认值: 128。
            num_layers_edge_encoder (Union[int, None], optional): 边编码器的层数。如果为 None，则不使用编码器。默认值: 2。
            hidden_dim_node_decoder (int, optional): 节点解码器的隐层维度。默认值: 128。
            num_layers_node_decoder (Union[int, None], optional): 节点解码器的层数。如果为 None，则不使用解码器。默认值: 2。
            aggregation (str, optional): 消息聚合方式，可选 "sum", "mean" 等。默认值: "sum"。
            do_concat_trick (bool, optional): 是否使用拼接优化技巧 (MLP+idx+sum) 以节省显存。默认值: False。
            num_processor_checkpoint_segments (int, optional): 梯度检查点 (Gradient Checkpointing) 的分段数。0 表示禁用。默认值: 0。
            recompute_activation (bool, optional): 是否重计算激活函数以节省显存。默认值: False。

        形状:
            输入 node_features: (N, input_dim_nodes)，其中 N 为节点总数。
            输入 edge_features: (M, input_dim_edges)，其中 M 为边总数。
            输入 graph: DGLGraph 或 CuGraphCSC，定义图拓扑结构。
            输出: (N, output_dim)，解码后的节点物理量。

    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        mlp_activation_fn: Union[str, List[str]] = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        recompute_activation: bool = False,
    ):
        super().__init__(meta=MetaData())

        activation_fn = get_activation(mlp_activation_fn)

        # 1. Edge Encoder (OneMlp)
        self.edge_encoder = OneMlp(
            style="MeshGraphMLP",
            input_dim=input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=recompute_activation,
        )

        # 2. Node Encoder (OneMlp)
        self.node_encoder = OneMlp(
            style="MeshGraphMLP",
            input_dim=input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=recompute_activation,
        )

        # 3. Node Decoder (OneMlp)
        self.node_decoder = OneMlp(
            style="MeshGraphMLP",
            input_dim=hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

        # 4. Processor (Core GNN)
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
    ) -> Tensor:
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph)
        x = self.node_decoder(x)
        return x


class MeshGraphNetProcessor(nn.Module):
    """
        MeshGraphNet 核心处理器 (Processor)。

        该模块由一系列堆叠的消息传递块 (Message Passing Blocks) 组成。
        每个块包含两个步骤：
        1. **Edge Block**: 使用 OneEdge (如 MeshEdgeBlock) 更新边特征。
        2. **Node Block**: 使用 OneNode (如 MeshNodeBlock) 聚合边信息并更新节点特征。

        支持梯度检查点 (Gradient Checkpointing) 以减少大规模图训练时的显存占用。

        Args:
            processor_size (int, optional): 处理器包含的消息传递层数。默认值: 15。
            input_dim_node (int, optional): 输入节点特征维度。默认值: 128。
            input_dim_edge (int, optional): 输入边特征维度。默认值: 128。
            num_layers_node (int, optional): 节点更新 MLP 的层数。默认值: 2。
            num_layers_edge (int, optional): 边更新 MLP 的层数。默认值: 2。
            aggregation (str, optional): 消息聚合方式 ("sum", "mean" 等)。默认值: "sum"。
            norm_type (str, optional): 归一化类型。默认值: "LayerNorm"。
            activation_fn (nn.Module, optional): 激活函数。默认值: nn.ReLU()。
            do_concat_trick (bool, optional): 是否启用显存优化技巧。默认值: False。
            num_processor_checkpoint_segments (int, optional): 梯度检查点分段数。默认值: 0 (禁用)。

        形状:
            输入 node_features: (N, input_dim_node)
            输入 edge_features: (M, input_dim_edge)
            输入 graph: DGLGraph
            输出: (N, input_dim_node) - 仅返回更新后的节点特征。
    
    """

    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        activation_fn: nn.Module = nn.ReLU(),
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.num_processor_checkpoint_segments = num_processor_checkpoint_segments

        edge_blocks = []
        node_blocks = []

        for _ in range(self.processor_size):
            edge_blocks.append(
                OneEdge(
                    style="MeshEdgeBlock",
                    input_dim_nodes=input_dim_node,
                    input_dim_edges=input_dim_edge,
                    output_dim=input_dim_edge,
                    hidden_dim=input_dim_edge,
                    hidden_layers=num_layers_edge,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                    do_concat_trick=do_concat_trick,
                    recompute_activation=False
                )
            )
            # 使用 OneNode 工厂
            node_blocks.append(
                OneNode(
                    style="MeshNodeBlock",
                    aggregation=aggregation,
                    input_dim_nodes=input_dim_node,
                    input_dim_edges=input_dim_edge,
                    output_dim=input_dim_node,
                    hidden_dim=input_dim_node,
                    hidden_layers=num_layers_node,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                    recompute_activation=False
                )
            )

        # 按照 Edge -> Node 的顺序交替排列
        layers = list(chain(*zip(edge_blocks, node_blocks)))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        self.set_checkpoint_segments(self.num_processor_checkpoint_segments)

    def set_checkpoint_segments(self, checkpoint_segments: int):
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError(
                    "Processor layers must be a multiple of checkpoint_segments"
                )
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    def run_function(
        self, segment_start: int, segment_end: int
    ) -> Callable[
        [Tensor, Tensor, Union[DGLGraph, List[DGLGraph]]], Tuple[Tensor, Tensor]
    ]:
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(
            node_features: Tensor,
            edge_features: Tensor,
            graph: Union[DGLGraph, List[DGLGraph]],
        ) -> Tuple[Tensor, Tensor]:
            for module in segment:
                edge_features, node_features = module(
                    edge_features, node_features, graph
                )
            return edge_features, node_features

        return custom_forward

    @torch.jit.unused
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
    ) -> Tensor:
        for segment_start, segment_end in self.checkpoint_segments:
            edge_features, node_features = self.checkpoint_fn(
                self.run_function(segment_start, segment_end),
                node_features,
                edge_features,
                graph,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        return node_features