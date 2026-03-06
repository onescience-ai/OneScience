import torch
from torch import nn

from .xihetransformer import XiHeTransformer3D
from .preln_transformer_block import PreLNTransformerBlock
from .factformer_block import Factformer_block
from .galerkin_transformer_block import Galerkin_Transformer_block
from .gnot_transformer_block import GNOTTransformerBlock
from .Neural_Spectral_Block import NeuralSpectralBlock1D, NeuralSpectralBlock2D, NeuralSpectralBlock3D
from .orthogonal_neural_block import OrthogonalNeuralBlock
from .SwinTransformerBlock import SwinTransformerBlock
from .Transolver_block import Transolver_block
# 构建统一的 Transformer 注册表
_TRANSFORMER_REGISTRY = {
    "XiHeTransformer3D": XiHeTransformer3D,
    "PreLNTransformerBlock": PreLNTransformerBlock,
    "Factformer_block": Factformer_block,
    "Galerkin_Transformer_block": Galerkin_Transformer_block,
    "GNOTTransformerBlock": GNOTTransformerBlock,
    "NeuralSpectralBlock1D": NeuralSpectralBlock1D,
    "NeuralSpectralBlock2D": NeuralSpectralBlock2D,
    "NeuralSpectralBlock3D": NeuralSpectralBlock3D,
    "OrthogonalNeuralBlock": OrthogonalNeuralBlock,
    "SwinTransformerBlock": SwinTransformerBlock,
    "Transolver_block": Transolver_block,
}

class OneTransformer(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _TRANSFORMER_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_TRANSFORMER_REGISTRY.keys())}"
            )
        
        self.transformer_block = _TRANSFORMER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        """
        前向传播。
        """
        return self.transformer_block(*args, **kwargs)