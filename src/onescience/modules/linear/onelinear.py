from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_LINEAR_REGISTRY = {
    "ProtenixLinear": ("onescience.modules.linear.protenixlinear", "ProtenixLinear"),
    "ProtenixLinearNoBias": (
        "onescience.modules.linear.protenixlinear",
        "ProtenixLinearNoBias",
    ),
    "ProtenixBiasInitLinear": (
        "onescience.modules.linear.protenixlinear",
        "ProtenixBiasInitLinear",
    ),
}


class OneLinear(nn.Module):
    """OneLinear module for linear operations with various styles."""

    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.Linear = instantiate_registered_style(style, _LINEAR_REGISTRY, "linear", **kwargs)

    def forward(self, *args, **kwargs):
        return self.Linear(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.Linear, name)
