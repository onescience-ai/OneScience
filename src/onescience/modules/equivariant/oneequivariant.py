import torch.nn as nn

from .group_conv import GroupEquivariantConv2d, GroupEquivariantConv3d

try:
    from .mace_interaction_blocks import (
        EquivariantProductBasisBlock as MaceEquivariantProductBasisBlock,
        InteractionBlock as MaceInteractionBlock,
        RealAgnosticAttResidualInteractionBlock as MaceRealAgnosticAttResidualInteractionBlock,
        RealAgnosticDensityInteractionBlock as MaceRealAgnosticDensityInteractionBlock,
        RealAgnosticDensityResidualInteractionBlock as MaceRealAgnosticDensityResidualInteractionBlock,
        RealAgnosticInteractionBlock as MaceRealAgnosticInteractionBlock,
        RealAgnosticResidualInteractionBlock as MaceRealAgnosticResidualInteractionBlock,
    )
except Exception:  # pragma: no cover - optional MACE deps
    MaceEquivariantProductBasisBlock = None
    MaceInteractionBlock = None
    MaceRealAgnosticAttResidualInteractionBlock = None
    MaceRealAgnosticDensityInteractionBlock = None
    MaceRealAgnosticDensityResidualInteractionBlock = None
    MaceRealAgnosticInteractionBlock = None
    MaceRealAgnosticResidualInteractionBlock = None

try:
    from .uma_so2_layers import SO2_Convolution as UmaSO2Convolution
    from .uma_so3_layers import SO3_Linear as UmaSO3Linear
except Exception:  # pragma: no cover - optional UMA deps
    UmaSO2Convolution = None
    UmaSO3Linear = None

_EQUIVARIANT_REGISTRY = {
    "GroupEquivariantConv2d": GroupEquivariantConv2d,
    "GroupEquivariantConv3d": GroupEquivariantConv3d,
}

if MaceInteractionBlock is not None:
    _EQUIVARIANT_REGISTRY.update(
        {
            "MaceInteractionBlock": MaceInteractionBlock,
            "MaceRealAgnosticInteractionBlock": MaceRealAgnosticInteractionBlock,
            "MaceRealAgnosticResidualInteractionBlock": MaceRealAgnosticResidualInteractionBlock,
            "MaceRealAgnosticDensityInteractionBlock": MaceRealAgnosticDensityInteractionBlock,
            "MaceRealAgnosticDensityResidualInteractionBlock": MaceRealAgnosticDensityResidualInteractionBlock,
            "MaceRealAgnosticAttResidualInteractionBlock": MaceRealAgnosticAttResidualInteractionBlock,
            "MaceEquivariantProductBasisBlock": MaceEquivariantProductBasisBlock,
        }
    )

if UmaSO2Convolution is not None:
    _EQUIVARIANT_REGISTRY.update(
        {
            "UmaSO2Convolution": UmaSO2Convolution,
            "UmaSO3Linear": UmaSO3Linear,
        }
    )


class OneEquivariant(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _EQUIVARIANT_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_EQUIVARIANT_REGISTRY.keys())}"
            )

        self.equivariant_layer = _EQUIVARIANT_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.equivariant_layer(*args, **kwargs)
