import torch.nn as nn
# from .protenixmsa import ProtenixMSAModule

_MSA_REGISTRY = {
    # "ProtenixMSAModule": ProtenixMSAModule,
}


class OneMSA(nn.Module):
    """OneMSA module for msa interface"""
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _MSA_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.msa = _MSA_REGISTRY[style](**kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.msa, name)

    def forward(self, *args, **kwargs):
        return self.msa(*args, **kwargs)

