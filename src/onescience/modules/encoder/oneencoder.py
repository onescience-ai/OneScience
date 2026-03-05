# from .timestepembedder import TimestepEmbedder 
from torch import nn
from .fengwuencoder import FengWuEncoder
from .graphvitencoder import GraphViTEncoder
from .meshgraphencoder import MeshGraphEncoder

_ENCODER_REGISTRY = {
    "FengWuEncoder": FengWuEncoder,
    "GraphViTEncoder": GraphViTEncoder,
    "MeshGraphEncoder":MeshGraphEncoder,
}

class OneEncoder(nn.Module):
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _ENCODER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.encoder = _ENCODER_REGISTRY[style](**kwargs)

    def forward(self, x):
        
        return self.encoder(x) 