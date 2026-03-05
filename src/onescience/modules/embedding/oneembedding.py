# from .timestepembedder import TimestepEmbedder
from torch import nn

from .panguembedding2d import PanguEmbedding2D
from .panguembedding3d import PanguEmbedding3D
from .protenixembedding import (
    ProtenixFourierEmbedding,
    ProtenixInputFeatureEmbedder,
    ProtenixTemplateEmbedder,
)

_EMBEDDER_REGISTRY = {
    "PanguEmbedding2D": PanguEmbedding2D,
    "PanguEmbedding3D": PanguEmbedding3D,
    "ProtenixFourierEmbedding": ProtenixFourierEmbedding,
    "ProtenixInputFeatureEmbedder": ProtenixInputFeatureEmbedder,
    "ProtenixTemplateEmbedder": ProtenixTemplateEmbedder,
    # "TimestepEmbedder": TimestepEmbedder,
}

class OneEmbedding(nn.Module):

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _EMBEDDER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.embedder = _EMBEDDER_REGISTRY[style](**kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.embedder, name)

    def forward(self, *args, **kwargs):
        return self.embedder(*args, **kwargs) 