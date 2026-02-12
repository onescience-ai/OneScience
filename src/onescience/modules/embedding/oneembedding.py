import torch
from torch import nn

from timestepembedder import TimestepEmbedder 
from panguembedding2d import PanguEmbedding2D
from panguembedding3d import PanguEmbedding3D
from timestep_embedding import timestep_embedding
from unified_pos_embedding import unified_pos_embedding

_EMBEDDER_REGISTRY = {
    "TimestepEmbedder": TimestepEmbedder,
    "PanguEmbedding2D": PanguEmbedding2D,
    "PanguEmbedding3D": PanguEmbedding3D,
    "timestep_embedding": timestep_embedding
    "unified_pos_embedding": unified_pos_embedding
}

class OneEmbedding(nn.Module):
   
    def __inin__(self, style: str, **kwargs):
        super().__init__()

        if style not in _EMBEDDER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.style = style
        self.embedder = _EMBEDDER_REGISTRY[style](**kwargs)

    def forward(self, x):
        
        return self.embedder(x) 