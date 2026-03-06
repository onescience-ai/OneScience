# from .timestepembedder import TimestepEmbedder 
from torch import nn

from .panguembedding2d import PanguEmbedding2D
from .panguembedding3d import PanguEmbedding3D
from .fuxiembedding import FuxiEmbedding
from .fourcastnetembedding import FourCastNetEmbedding
from .xiheembedding import XiheEmbedding
from .graphcast_embedder import GraphCastEncoderEmbedder, GraphCastDecoderEmbedder

_EMBEDDER_REGISTRY = {
    "PanguEmbedding2D": PanguEmbedding2D,
    "PanguEmbedding3D": PanguEmbedding3D,
    "FuxiEmbedding": FuxiEmbedding,
    "FourCastNetEmbedding": FourCastNetEmbedding,
    "XiheEmbedding":XiheEmbedding,
    "GraphCastEncoderEmbedder": GraphCastEncoderEmbedder,
    "GraphCastDecoderEmbedder": GraphCastDecoderEmbedder,
}

class OneEmbedding(nn.Module):
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _EMBEDDER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.embedder = _EMBEDDER_REGISTRY[style](**kwargs)

    def forward(self, x):
        
        return self.embedder(x) 