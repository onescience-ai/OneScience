import torch.nn as nn
from onescience.modules._lazy import instantiate_registered_style

_MSA_REGISTRY = {
    "ProtenixMSAModule": ("onescience.modules.msa.protenixmsa", "ProtenixMSAModule"),
}


class OneMSA(nn.Module):
    """OneMSA module for msa interface"""
    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.msa = instantiate_registered_style(style, _MSA_REGISTRY, "msa", **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.msa, name)

    def forward(self, *args, **kwargs):
        return self.msa(*args, **kwargs)
