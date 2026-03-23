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
from .fuxitransformer import FuxiTransformer
from .earthtransformer2Dblock import EarthTransformer2DBlock
from .earthtransformer3Dblock import EarthTransformer3DBlock
from .earthdistributedtransformer3Dblock import EarthDistributedTransformer3DBlock
from .xihelocaltransformer import XihelocalTransformer
from .protenixtransformer import (
     ProtenixConditionedTransitionBlock,
     ProtenixDiffusionTransformerBlock,
     ProtenixDiffusionTransformer,
     ProtenixAtomTransformer,
 )

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
    "FuxiTransformer": FuxiTransformer,
    "EarthTransformer2DBlock": EarthTransformer2DBlock,
    "EarthTransformer3DBlock": EarthTransformer3DBlock,
    "EarthDistributedTransformer3DBlock": EarthDistributedTransformer3DBlock,
    "XihelocalTransformer": XihelocalTransformer,
    "ProtenixConditionedTransitionBlock": ProtenixConditionedTransitionBlock,
    "ProtenixDiffusionTransformerBlock": ProtenixDiffusionTransformerBlock,
    "ProtenixDiffusionTransformer": ProtenixDiffusionTransformer,
    "ProtenixAtomTransformer": ProtenixAtomTransformer,
}

class OneTransformer(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _TRANSFORMER_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_TRANSFORMER_REGISTRY.keys())}"
            )
        
        self.transformer = _TRANSFORMER_REGISTRY[style](**kwargs)
        
    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)



