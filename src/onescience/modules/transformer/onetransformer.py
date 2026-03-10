from torch import nn

from .fuxitransformer import FuxiTransformer
from .earthtransformer2Dblock import EarthTransformer2DBlock
from .earthtransformer3Dblock import EarthTransformer3DBlock
from .earthdistributedtransformer3Dblock import EarthDistributedTransformer3DBlock
from .xihelocaltransformer import XihelocalTransformer
from .xihetransformer import XiHeTransformer3D
# from .protenixtransformer import (
#     ProtenixConditionedTransitionBlock,
#     ProtenixDiffusionTransformerBlock,
#     ProtenixDiffusionTransformer,
#     ProtenixAtomTransformer,
# )

_TRANSFORMER_REGISTRY = {
    "FuxiTransformer": FuxiTransformer,
    "EarthTransformer2DBlock": EarthTransformer2DBlock,
    "EarthTransformer3DBlock": EarthTransformer3DBlock,
    "EarthDistributedTransformer3DBlock": EarthDistributedTransformer3DBlock,
    "XihelocalTransformer": XihelocalTransformer,
    "XiHeTransformer3D": XiHeTransformer3D,
    # "ProtenixConditionedTransitionBlock": ProtenixConditionedTransitionBlock,
    # "ProtenixDiffusionTransformerBlock": ProtenixDiffusionTransformerBlock,
    # "ProtenixDiffusionTransformer": ProtenixDiffusionTransformer,
    # "ProtenixAtomTransformer": ProtenixAtomTransformer,
}

class OneTransformer(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _TRANSFORMER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.transformer = _TRANSFORMER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

