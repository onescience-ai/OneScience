from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_PAIRFORMER_REGISTRY = {
    "ProtenixPairformerBlock": (
        "onescience.modules.pairformer.protenixpairformer",
        "ProtenixPairformerBlock",
    ),
    "ProtenixPairformerStack": (
        "onescience.modules.pairformer.protenixpairformer",
        "ProtenixPairformerStack",
    ),
}


class OnePairformer(nn.Module):
    """OnePairformer module for pairformer operations with various styles."""

    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.Pairformer = instantiate_registered_style(
            style,
            _PAIRFORMER_REGISTRY,
            "pairformer",
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        return self.Pairformer(*args, **kwargs)


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.Pairformer, name)
