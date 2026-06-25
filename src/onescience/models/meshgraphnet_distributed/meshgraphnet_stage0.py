"""
MeshGraphNet Stage 0: Encoder

This stage encodes node and edge features into high-dimensional latent space.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dgl import DGLGraph

import onescience
from onescience.modules import OneMlp
from onescience.modules.mlp.onemlp import _MLP_REGISTRY
from onescience.modules.layer.activations import get_activation


class MeshGraphNetStage0(nn.Module):
    """
    MeshGraphNet Stage 0: Encoder

    This stage encodes node and edge features into high-dimensional latent space.
    It does not read from input_tensor (first stage in pipeline).

    Args:
        input_dim_nodes (int): Input node feature dimension
        input_dim_edges (int): Input edge feature dimension
        hidden_dim_processor (int): Hidden dimension for processor
        hidden_dim_node_encoder (int): Hidden dimension for node encoder
        num_layers_node_encoder (int): Number of layers in node encoder
        hidden_dim_edge_encoder (int): Hidden dimension for edge encoder
        num_layers_edge_encoder (int): Number of layers in edge encoder
        mlp_activation_fn (str): Activation function for MLP
        recompute_activation (bool): Whether to recompute activation
        config: Megatron config object (for tensor parallel)
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        hidden_dim_processor: int,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        mlp_activation_fn: str = "relu",
        recompute_activation: bool = False,
        config=None,
    ):
        super().__init__()

        activation_fn = get_activation(mlp_activation_fn)

        # Determine if using distributed MLP
        use_distributed = config and config.tensor_model_parallel_size > 1
        mlp_style = "MeshGraphDistributedMLP" if use_distributed else "MeshGraphMLP"

        # Factory function for creating MLP
        def _create_mlp(style, config, **kwargs):
            if style == "MeshGraphDistributedMLP":
                return _MLP_REGISTRY[style](config=config, **kwargs)
            else:
                return _MLP_REGISTRY[style](**kwargs)

        # Edge Encoder
        edge_encoder_kwargs = {
            "input_dim": input_dim_edges,
            "output_dim": hidden_dim_processor,
            "hidden_dim": hidden_dim_edge_encoder,
            "hidden_layers": num_layers_edge_encoder,
            "activation_fn": activation_fn,
            "norm_type": "LayerNorm",
            "recompute_activation": recompute_activation,
        }
        self.edge_encoder = _create_mlp(mlp_style, config, **edge_encoder_kwargs)

        # Node Encoder
        node_encoder_kwargs = {
            "input_dim": input_dim_nodes,
            "output_dim": hidden_dim_processor,
            "hidden_dim": hidden_dim_node_encoder,
            "hidden_layers": num_layers_node_encoder,
            "activation_fn": activation_fn,
            "norm_type": "LayerNorm",
            "recompute_activation": recompute_activation,
        }
        self.node_encoder = _create_mlp(mlp_style, config, **node_encoder_kwargs)

    def set_input_tensor(self, input_tensor):
        """Megatron pipeline scheduling hook (not used in Stage 0)"""
        self.input_tensor = input_tensor

    def forward(self, node_features, edge_features) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Stage 0

        Args:
            node_features: Node features (num_nodes, input_dim_nodes)
            edge_features: Edge features (num_edges, input_dim_edges)

        Returns:
            Tuple of (node_features, edge_features) for next stage
        """
        # Encode features
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)

        return node_features, edge_features