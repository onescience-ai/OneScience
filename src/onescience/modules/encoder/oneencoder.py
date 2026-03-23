import torch
from torch import nn

from .unet_encoder import UNetEncoder1D, UNetEncoder2D, UNetEncoder3D
from .graphvit_encoder import GraphViTEncoder
from .mesh_graph_encoder import MeshGraphEncoder
from .fengwuencoder import FengWuEncoder
from .protenixencoding import (
     ProtenixRelativePositionEncoding,
     ProtenixAtomAttentionEncoder,
     )

_ENCODER_REGISTRY = {
    "UNetEncoder1D": UNetEncoder1D,
    "UNetEncoder2D": UNetEncoder2D,
    "UNetEncoder3D": UNetEncoder3D,
    "GraphViTEncoder": GraphViTEncoder,
    "MeshGraphEncoder":MeshGraphEncoder,
    "FengWuEncoder": FengWuEncoder,
    "ProtenixRelativePositionEncoding": ProtenixRelativePositionEncoding,
    "ProtenixAtomAttentionEncoder": ProtenixAtomAttentionEncoder,
}

class OneEncoder(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _ENCODER_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_ENCODER_REGISTRY.keys())}"
            )
        self.encoder = _ENCODER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        new_state = {'encoder.' + k: v for k, v in state_dict.items()}
        return super().load_state_dict(new_state, strict=strict)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)

