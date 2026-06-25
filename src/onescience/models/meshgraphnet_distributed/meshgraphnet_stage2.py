"""
MeshGraphNet Stage 2: Decoder

This stage decodes processed node features back to physical space.
"""

import torch
import torch.nn as nn
from typing import Tuple

import onescience
from onescience.modules import OneMlp
from onescience.modules.mlp.onemlp import _MLP_REGISTRY
from onescience.modules.layer.activations import get_activation


class MeshGraphNetStage2(nn.Module):
    """
    MeshGraphNet Stage 2: Decoder

    This stage decodes processed node features back to physical space.
    It reads from input_tensor and returns final output.

    Args:
        output_dim (int): Output feature dimension
        hidden_dim_processor (int): Hidden dimension for processor
        hidden_dim_node_decoder (int): Hidden dimension for node decoder
        num_layers_node_decoder (int): Number of layers in node decoder
        mlp_activation_fn (str): Activation function for MLP
        recompute_activation (bool): Whether to recompute activation
        config: Megatron config object (for tensor parallel)
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim_processor: int,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        mlp_activation_fn: str = "relu",
        recompute_activation: bool = False,
        config=None,
    ):
        super().__init__()
        self.input_tensor = None

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

        # Node Decoder
        mlp_kwargs = {
            "input_dim": hidden_dim_processor,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim_node_decoder,
            "hidden_layers": num_layers_node_decoder,
            "activation_fn": activation_fn,
            "norm_type": None,
            "recompute_activation": recompute_activation,
        }
        self.node_decoder = _create_mlp(mlp_style, config, **mlp_kwargs)

    def set_input_tensor(self, input_tensor):
        """Megatron pipeline scheduling hook"""
        self.input_tensor = input_tensor

    def forward(self, node_features, edge_features) -> torch.Tensor:
        """
        Args:
            node_features: Node features (num_nodes, hidden_dim)
            edge_features: Edge features (num_edges, hidden_dim)

        Returns:
            Decoded node features (num_nodes, output_dim)
        """
        # Decode node features
        output = self.node_decoder(node_features)

        return output