"""Compatibility re-export for MACE blocks."""

import torch

from onescience.modules.embedding.mace_embedding_blocks import (
    AtomicEnergiesBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
)
from onescience.modules.equivariant.mace_interaction_blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from onescience.modules.head.mace_readout_blocks import (
    LinearDipoleReadoutBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    ScaleShiftBlock,
)

nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}

__all__ = [
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "NonLinearReadoutBlock",
    "LinearDipoleReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "EquivariantProductBasisBlock",
    "InteractionBlock",
    "RealAgnosticInteractionBlock",
    "RealAgnosticResidualInteractionBlock",
    "RealAgnosticDensityInteractionBlock",
    "RealAgnosticDensityResidualInteractionBlock",
    "RealAgnosticAttResidualInteractionBlock",
    "ScaleShiftBlock",
    "nonlinearities",
]
