from torch import nn

from .protenixlinear import (
    ProtenixLinear,
    ProtenixLinearNoBias,
    ProtenixBiasInitLinear,
)

_LINEAR_REGISTRY = {
    "ProtenixLinear": ProtenixLinear,
    "ProtenixLinearNoBias": ProtenixLinearNoBias,
    "ProtenixBiasInitLinear": ProtenixBiasInitLinear,
}


class OneLinear(nn.Module):
    """OneLinear module for linear operations with various styles."""

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _LINEAR_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.Linear = _LINEAR_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.Linear(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.Linear, name)
