"""
MeshGraphNet Stage 1: Processor

This stage performs message passing through multiple layers.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from itertools import chain

import onescience
from onescience.modules import OneEdge, OneNode
from onescience.modules.edge.oneedge import _EDGE_REGISTRY
from onescience.modules.node.onenode import _NODE_REGISTRY
from onescience.modules.layer.activations import get_activation


class MeshGraphNetStage1(nn.Module):
    """
    MeshGraphNet Stage 1: Processor

    This stage performs message passing through multiple layers.
    It reads from input_tensor and passes through meta tensor.

    Args:
        processor_size (int): Total number of message passing layers
        layer_start (int): Starting layer index (0-based)
        layer_end (int): Ending layer index (exclusive)
        input_dim_node (int): Input node feature dimension
        input_dim_edge (int): Input edge feature dimension
        num_layers_node (int): Number of layers in node MLP
        num_layers_edge (int): Number of layers in edge MLP
        aggregation (str): Aggregation method ("sum", "mean", etc.)
        norm_type (str): Normalization type
        mlp_activation_fn (str): Activation function for MLP
        do_concat_trick (bool): Whether to use concat trick
        recompute_activation (bool): Whether to recompute activation
        config: Megatron config object (for tensor parallel)
    """

    def __init__(
        self,
        processor_size: int = 15,
        layer_start: int = 0,
        layer_end: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        mlp_activation_fn: str = "relu",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
        config=None,
    ):
        super().__init__()
        self.input_tensor = None
        self.processor_size = processor_size
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.actual_processor_size = layer_end - layer_start

        activation_fn = get_activation(mlp_activation_fn)

        # Determine if using distributed modules
        use_distributed = config and config.tensor_model_parallel_size > 1
        edge_style = "MeshEdgeDistributedBlock" if use_distributed else "MeshEdgeBlock"
        node_style = "MeshNodeDistributedBlock" if use_distributed else "MeshNodeBlock"

        # Factory functions for creating blocks
        def _create_edge_block(style, config, **kwargs):
            if style == "MeshEdgeDistributedBlock":
                return _EDGE_REGISTRY[style](config=config, **kwargs)
            else:
                return _EDGE_REGISTRY[style](**kwargs)

        def _create_node_block(style, config, **kwargs):
            if style == "MeshNodeDistributedBlock":
                return _NODE_REGISTRY[style](config=config, **kwargs)
            else:
                return _NODE_REGISTRY[style](**kwargs)

        # Create edge and node blocks (only for specified layer range)
        edge_blocks = []
        node_blocks = []

        for i in range(layer_start, layer_end):
            edge_kwargs = {
                "input_dim_nodes": input_dim_node,
                "input_dim_edges": input_dim_edge,
                "output_dim": input_dim_edge,
                "hidden_dim": input_dim_edge,
                "hidden_layers": num_layers_edge,
                "activation_fn": activation_fn,
                "norm_type": norm_type,
                "do_concat_trick": do_concat_trick,
                "recompute_activation": recompute_activation,
            }
            edge_blocks.append(_create_edge_block(edge_style, config, **edge_kwargs))

            node_kwargs = {
                "aggregation": aggregation,
                "input_dim_nodes": input_dim_node,
                "input_dim_edges": input_dim_edge,
                "output_dim": input_dim_node,
                "hidden_dim": input_dim_node,
                "hidden_layers": num_layers_node,
                "activation_fn": activation_fn,
                "norm_type": norm_type,
                "recompute_activation": recompute_activation,
            }
            node_blocks.append(_create_node_block(node_style, config, **node_kwargs))

        # Interleave edge and node blocks
        layers = list(chain(*zip(edge_blocks, node_blocks)))
        self.processor_layers = nn.ModuleList(layers)

    def set_input_tensor(self, input_tensor):
        """Megatron pipeline scheduling hook"""
        self.input_tensor = input_tensor

    def forward(self, node_features, edge_features, graph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Stage 1

        Args:
            node_features: Node features (num_nodes, hidden_dim)
            edge_features: Edge features (num_edges, hidden_dim)
            graph: DGLGraph object

        Returns:
            Tuple of (node_features, edge_features) for next stage
        """
        # Process through message passing layers
        for layer in self.processor_layers:
            edge_features, node_features = layer(edge_features, node_features, graph)

        return node_features, edge_features