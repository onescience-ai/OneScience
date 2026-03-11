import torch
import torch.nn as nn
from torch import Tensor

try:
    import dgl
    from dgl import DGLGraph
except ImportError:
    pass 

from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Tuple, Union

# --- 引入模块工厂 ---
from onescience.modules import OneEdge, OneNode, OneMlp

# 保持工具类引用
from onescience.modules.utils.gnnlayer_utils import CuGraphCSC, set_checkpoint_fn
from onescience.modules.layer.activations import get_activation
from onescience.modules.meta import ModelMetaData
from onescience.modules.module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "MeshGraphNet"
    # Optimization
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


class Model(Module):
    """
    LSMMeshGraphNet 网络架构 (Refactored).
    
    使用 OneMlp, OneEdge, OneNode 工厂构建。
    """

    def __init__(
        self,
        args,
        device,
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
        self.__name__ = "LSMMeshGraphNet"
        
        # 参数绑定
        self.input_dim_nodes = args.fun_dim
        self.input_dim_edges = 4 
        self.output_dim = args.out_dim
        
        activation_fn = get_activation(mlp_activation_fn)

        # 1. Edge Encoder
        self.edge_encoder = OneMlp(
            style="MeshGraphMLP",
            input_dim=self.input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=recompute_activation,
        )

        # 2. Node Encoder
        self.node_encoder = OneMlp(
            style="MeshGraphMLP",
            input_dim=self.input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=recompute_activation,
        )

        # 3. Node Decoder
        self.node_decoder = OneMlp(
            style="MeshGraphMLP",
            input_dim=hidden_dim_processor,
            output_dim=self.output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

        # 4. Processor
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
    MeshGraphNet processor block constructed via OneEdge and OneNode factories.
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