from torch import nn

from .diffusionmodule import DiffusionModule
# from .protenixdiffusion import (
#     ProtenixDiffusionConditioning,
#     ProtenixDiffusionSchedule,
#     ProtenixDiffusionModule,
# )

_DIFFUSION_REGISTRY = {
    "DiffusionModule": DiffusionModule,
    # "ProtenixDiffusionConditioning": ProtenixDiffusionConditioning,
    # "ProtenixDiffusionSchedule": ProtenixDiffusionSchedule,
    # "ProtenixDiffusionModule": ProtenixDiffusionModule,
}


class OneDiffusion(nn.Module):
    """
    OneDiffusion module for diffusion operations with various styles.

    Supports:
    - DiffusionModule: Generic diffusion module
    - ProtenixDiffusionConditioning: Protenix conditioning (Algorithm 21)
    - ProtenixDiffusionSchedule: Protenix noise schedule
    - ProtenixDiffusionModule: Protenix diffusion module (Algorithm 20)
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _DIFFUSION_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.Diffusion = _DIFFUSION_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.Diffusion(*args, **kwargs)


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.Diffusion, name)