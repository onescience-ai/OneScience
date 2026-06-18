from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_DIFFUSION_REGISTRY = {
    "DiffusionModule": ("onescience.modules.diffusion.diffusionmodule", "DiffusionModule"),
    "ProtenixDiffusionConditioning": (
        "onescience.modules.diffusion.protenixdiffusion",
        "ProtenixDiffusionConditioning",
    ),
    "ProtenixDiffusionSchedule": (
        "onescience.modules.diffusion.protenixdiffusion",
        "ProtenixDiffusionSchedule",
    ),
    "ProtenixDiffusionModule": (
        "onescience.modules.diffusion.protenixdiffusion",
        "ProtenixDiffusionModule",
    ),
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

        self.Diffusion = instantiate_registered_style(
            style,
            _DIFFUSION_REGISTRY,
            "diffusion",
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        return self.Diffusion(*args, **kwargs)


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.Diffusion, name)
