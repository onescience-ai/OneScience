import torch
from torch import nn

from .mesh_graph_mlp import MeshGraphMLP
from .mesh_graph_mlp import MeshGraphEdgeMLPConcat
from .MLP import StandardMLP, SimpleMLP, DeepResMLP, RegularizedMLP, LightweightMLP
from .GMLP import GroupEquivariantMLP2d, GroupEquivariantMLP3d

_MLP_REGISTRY = {
    "MeshGraphMLP": MeshGraphMLP,
    "MeshGraphEdgeMLPConcat": MeshGraphEdgeMLPConcat,
    "StandardMLP": StandardMLP,
    "SimpleMLP": SimpleMLP,
    "DeepResMLP": DeepResMLP,
    "RegularizedMLP": RegularizedMLP,
    "LightweightMLP": LightweightMLP,
    "GroupEquivariantMLP2d": GroupEquivariantMLP2d,
    "GroupEquivariantMLP3d": GroupEquivariantMLP3d,
}

class OneMlp(nn.Module):
    """OneMlp module for MLP operations."""
    
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _MLP_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        # 使用 **kwargs 动态接收参数，避免硬编码
        self.mlp = _MLP_REGISTRY[style](**kwargs)
        
    def forward(self, x):
        return self.mlp(x)