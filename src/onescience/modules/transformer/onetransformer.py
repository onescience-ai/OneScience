from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_TRANSFORMER_REGISTRY = {
    "XiHeTransformer3D": ("onescience.modules.transformer.xihetransformer", "XiHeTransformer3D"),
    "PreLNTransformerBlock": (
        "onescience.modules.transformer.preln_transformer_block",
        "PreLNTransformerBlock",
    ),
    "Factformer_block": ("onescience.modules.transformer.factformer_block", "Factformer_block"),
    "Galerkin_Transformer_block": (
        "onescience.modules.transformer.galerkin_transformer_block",
        "Galerkin_Transformer_block",
    ),
    "GNOTTransformerBlock": (
        "onescience.modules.transformer.gnot_transformer_block",
        "GNOTTransformerBlock",
    ),
    "NeuralSpectralBlock1D": (
        "onescience.modules.transformer.Neural_Spectral_Block",
        "NeuralSpectralBlock1D",
    ),
    "NeuralSpectralBlock2D": (
        "onescience.modules.transformer.Neural_Spectral_Block",
        "NeuralSpectralBlock2D",
    ),
    "NeuralSpectralBlock3D": (
        "onescience.modules.transformer.Neural_Spectral_Block",
        "NeuralSpectralBlock3D",
    ),
    "OrthogonalNeuralBlock": (
        "onescience.modules.transformer.orthogonal_neural_block",
        "OrthogonalNeuralBlock",
    ),
    "SwinTransformerBlock": (
        "onescience.modules.transformer.SwinTransformerBlock",
        "SwinTransformerBlock",
    ),
    "Transolver_block": (
        "onescience.modules.transformer.Transolver_block",
        "Transolver_block",
    ),
    "FuxiTransformer": ("onescience.modules.transformer.fuxitransformer", "FuxiTransformer"),
    "EarthTransformer2DBlock": (
        "onescience.modules.transformer.earthtransformer2Dblock",
        "EarthTransformer2DBlock",
    ),
    "EarthTransformer3DBlock": (
        "onescience.modules.transformer.earthtransformer3Dblock",
        "EarthTransformer3DBlock",
    ),
    "EarthDistributedTransformer3DBlock": (
        "onescience.modules.transformer.earthdistributedtransformer3Dblock",
        "EarthDistributedTransformer3DBlock",
    ),
    "XihelocalTransformer": (
        "onescience.modules.transformer.xihelocaltransformer",
        "XihelocalTransformer",
    ),
    "XiheDistributedLocalTransformer": (
        "onescience.modules.transformer.xihedistributedlocaltransformer",
        "XiheDistributedLocalTransformer",
    ),
    "ProtenixConditionedTransitionBlock": (
        "onescience.modules.transformer.protenixtransformer",
        "ProtenixConditionedTransitionBlock",
    ),
    "ProtenixDiffusionTransformerBlock": (
        "onescience.modules.transformer.protenixtransformer",
        "ProtenixDiffusionTransformerBlock",
    ),
    "ProtenixDiffusionTransformer": (
        "onescience.modules.transformer.protenixtransformer",
        "ProtenixDiffusionTransformer",
    ),
    "ProtenixAtomTransformer": (
        "onescience.modules.transformer.protenixtransformer",
        "ProtenixAtomTransformer",
    ),
}

class OneTransformer(nn.Module):
    """
    Transformer 统一入口。

    通过 `style` 从注册表中选择具体实现，调用层无需直接 import 底层模块。
    当前天气相关模型中，常用实现包括：

    - `FuxiTransformer`
    - `EarthTransformer2DBlock`
    - `EarthTransformer3DBlock`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.transformer = instantiate_registered_style(
            style,
            _TRANSFORMER_REGISTRY,
            "transformer",
            **kwargs,
        )
        
    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)
