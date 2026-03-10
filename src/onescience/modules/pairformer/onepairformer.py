from torch import nn

# from .protenixpairformer import ProtenixPairformerBlock, ProtenixPairformerStack

_PAIRFORMER_REGISTRY = {
    # "ProtenixPairformerBlock": ProtenixPairformerBlock,
    # "ProtenixPairformerStack": ProtenixPairformerStack,
}


class OnePairformer(nn.Module):
    """OnePairformer module for pairformer operations with various styles."""

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _PAIRFORMER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.Pairformer = _PAIRFORMER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.Pairformer(*args, **kwargs)


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.Pairformer, name)