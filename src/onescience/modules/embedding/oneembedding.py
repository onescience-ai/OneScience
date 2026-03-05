from torch import nn

from .panguembedding2d import PanguEmbedding2D
from .panguembedding3d import PanguEmbedding3D
from .fourier_pos_embedding import FourierPosEmbedding
from .graphcast_embedder import GraphCastEncoderEmbedder, GraphCastDecoderEmbedder
_EMBEDDER_REGISTRY = {
    "PanguEmbedding2D": PanguEmbedding2D,
    "PanguEmbedding3D": PanguEmbedding3D,
    "FourierPosEmbedding": FourierPosEmbedding,
    "GraphCastEncoderEmbedder": GraphCastEncoderEmbedder,
    "GraphCastDecoderEmbedder": GraphCastDecoderEmbedder,
    # "TimestepEmbedder": TimestepEmbedder,
}

class OneEmbedding(nn.Module):
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _EMBEDDER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.embedder = _EMBEDDER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        
        return self.embedder(*args, **kwargs) 