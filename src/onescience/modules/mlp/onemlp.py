from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_MLP_REGISTRY = {
    "MeshGraphMLP": ("onescience.modules.mlp.mesh_graph_mlp", "MeshGraphMLP"),
    "MeshGraphEdgeMLPConcat": (
        "onescience.modules.mlp.mesh_graph_mlp",
        "MeshGraphEdgeMLPConcat",
    ),
    "MeshGraphEdgeMLPSum": ("onescience.modules.mlp.mesh_graph_mlp", "MeshGraphEdgeMLPSum"),
    "MeshGraphDistributedMLP": ("onescience.modules.mlp.mesh_graph_distributed_mlp", "MeshGraphDistributedMLP"),
    "StandardMLP": ("onescience.modules.mlp.MLP", "StandardMLP"),
    "SimpleMLP": ("onescience.modules.mlp.MLP", "SimpleMLP"),
    "DeepResMLP": ("onescience.modules.mlp.MLP", "DeepResMLP"),
    "RegularizedMLP": ("onescience.modules.mlp.MLP", "RegularizedMLP"),
    "LightweightMLP": ("onescience.modules.mlp.MLP", "LightweightMLP"),
    "GroupEquivariantMLP2d": ("onescience.modules.mlp.GMLP", "GroupEquivariantMLP2d"),
    "GroupEquivariantMLP3d": ("onescience.modules.mlp.GMLP", "GroupEquivariantMLP3d"),
    "XiheMlp": ("onescience.modules.mlp.xihemlp", "XiheMlp"),
    "XiheDistributedMlp": ("onescience.modules.mlp.xihedistributedmlp", "XiheDistributedMlp"),
}

class OneMlp(nn.Module):
    """OneMlp module for MLP operations."""
    
    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.mlp = instantiate_registered_style(style, _MLP_REGISTRY, "mlp", **kwargs)
        
    def forward(self, *args, **kwargs): 
        return self.mlp(*args, **kwargs)