from torch import nn

from .fourcastnetafno import FourCastNetAFNO2D

_AFNO_REGISTRY = {
    "FourCastNetAFNO2D": FourCastNetAFNO2D,
}

class OneAFNO(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _AFNO_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.afno = _AFNO_REGISTRY[style](**kwargs)

    def forward(self, x):
        return self.afno(x) 

    