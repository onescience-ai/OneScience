from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_EMBEDDER_REGISTRY = {
    "PanguEmbedding": ("onescience.modules.embedding.panguembedding", "PanguEmbedding"),
    "FourierPosEmbedding": (
        "onescience.modules.embedding.fourier_pos_embedding",
        "FourierPosEmbedding",
    ),
    "FuxiEmbedding": ("onescience.modules.embedding.fuxiembedding", "FuxiEmbedding"),
    "FourCastNetEmbedding": (
        "onescience.modules.embedding.fourcastnetembedding",
        "FourCastNetEmbedding",
    ),
    "XiheEmbedding": ("onescience.modules.embedding.xiheembedding", "XiheEmbedding"),
    "GraphCastEncoderEmbedder": (
        "onescience.modules.embedding.graphcast_embedder",
        "GraphCastEncoderEmbedder",
    ),
    "GraphCastDecoderEmbedder": (
        "onescience.modules.embedding.graphcast_embedder",
        "GraphCastDecoderEmbedder",
    ),
    "ProtenixFourierEmbedding": (
        "onescience.modules.embedding.protenixembedding",
        "ProtenixFourierEmbedding",
    ),
    "ProtenixInputFeatureEmbedder": (
        "onescience.modules.embedding.protenixembedding",
        "ProtenixInputFeatureEmbedder",
    ),
    "ProtenixTemplateEmbedder": (
        "onescience.modules.embedding.protenixembedding",
        "ProtenixTemplateEmbedder",
    ),
    "AtomTypeEmbedding": (
        "onescience.modules.embedding.matris_embedding",
        "AtomTypeEmbedding",
    ),
    "EdgeBasisEmbedding": (
        "onescience.modules.embedding.matris_embedding",
        "EdgeBasisEmbedding",
    ),
    "ThreebodyEmbedding": (
        "onescience.modules.embedding.matris_embedding",
        "ThreebodyEmbedding",
    ),
    "ThreebodyFourierExpansion": (
        "onescience.modules.embedding.matris_embedding",
        "ThreebodyFourierExpansion",
    ),
    "three_bodySHExpansion": (
        "onescience.modules.embedding.matris_embedding",
        "three_bodySHExpansion",
    ),
    "TimeEmbedding": (
        "onescience.modules.embedding.matris_embedding",
        "TimeEmbedding",
    ),
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

        self.embedder = instantiate_registered_style(
            style,
            _EMBEDDER_REGISTRY,
            "embedding",
            **kwargs,
        )
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.embedder, name)

    def forward(self, *args, **kwargs):
        return self.embedder(*args, **kwargs) 
