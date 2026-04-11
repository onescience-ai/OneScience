from torch import nn

from .panguembedding import PanguEmbedding
from .fourier_pos_embedding import FourierPosEmbedding
from .fuxiembedding import FuxiEmbedding
from .fourcastnetembedding import FourCastNetEmbedding
from .xiheembedding import XiheEmbedding
from .graphcast_embedder import GraphCastEncoderEmbedder, GraphCastDecoderEmbedder
from .protenixembedding import (
     ProtenixFourierEmbedding,
     ProtenixInputFeatureEmbedder,
     ProtenixTemplateEmbedder,
 )

_EMBEDDER_REGISTRY = {
    "PanguEmbedding": PanguEmbedding,
    "FourierPosEmbedding": FourierPosEmbedding,
    "FuxiEmbedding": FuxiEmbedding,
    "FourCastNetEmbedding": FourCastNetEmbedding,
    "XiheEmbedding": XiheEmbedding,
    "GraphCastEncoderEmbedder": GraphCastEncoderEmbedder,
    "GraphCastDecoderEmbedder": GraphCastDecoderEmbedder,
    "ProtenixFourierEmbedding": ProtenixFourierEmbedding,
    "ProtenixInputFeatureEmbedder": ProtenixInputFeatureEmbedder,
    "ProtenixTemplateEmbedder": ProtenixTemplateEmbedder,
}

class OneEmbedding(nn.Module):
    """
    Embedding 统一入口。

    通过 `style` 从注册表中选择具体 embedding 实现。
    当前天气相关模型中，常用实现包括：

    - `PanguEmbedding`
    - `FourCastNetEmbedding`
    - `FuxiEmbedding`
    """

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
