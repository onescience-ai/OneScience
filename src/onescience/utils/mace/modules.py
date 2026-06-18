"""Compatibility exports for legacy `onescience.utils.mace.modules` imports."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type

import torch

from onescience.models.mace.mace import (
    AtomicDipolesMACE,
    EnergyDipolesMACE,
    MACE,
    ScaleShiftMACE,
)
from onescience.modules.block.mace_block import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    ScaleShiftBlock,
)
from onescience.modules.equivariant.mace_symmetric_contraction import SymmetricContraction
from onescience.modules.func_utils.mace_func_utils import (
    compute_avg_num_neighbors,
    compute_fixed_charge_dipole,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_rms_dipoles,
    compute_statistics,
)
from onescience.modules.layer.mace_radial import (
    BesselBasis,
    GaussianBasis,
    PolynomialCutoff,
    ZBLBasis,
)
from onescience.modules.loss.mace_loss import (
    DipoleSingleLoss,
    UniversalLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesL1L2Loss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
)

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
}

scaling_classes: Dict[str, Callable] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
    "rms_dipoles_scaling": compute_rms_dipoles,
}

gate_dict: Dict[str, Optional[Callable]] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

__all__ = [
    "AtomicDipolesMACE",
    "AtomicEnergiesBlock",
    "BesselBasis",
    "DipoleSingleLoss",
    "EnergyDipolesMACE",
    "EquivariantProductBasisBlock",
    "GaussianBasis",
    "InteractionBlock",
    "LinearDipoleReadoutBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "MACE",
    "NonLinearDipoleReadoutBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "RadialEmbeddingBlock",
    "ScaleShiftBlock",
    "ScaleShiftMACE",
    "SymmetricContraction",
    "UniversalLoss",
    "WeightedEnergyForcesDipoleLoss",
    "WeightedEnergyForcesL1L2Loss",
    "WeightedEnergyForcesLoss",
    "WeightedEnergyForcesStressLoss",
    "WeightedEnergyForcesVirialsLoss",
    "WeightedForcesLoss",
    "WeightedHuberEnergyForcesStressLoss",
    "ZBLBasis",
    "compute_avg_num_neighbors",
    "compute_fixed_charge_dipole",
    "compute_mean_rms_energy_forces",
    "compute_mean_std_atomic_inter_energy",
    "compute_statistics",
    "gate_dict",
    "interaction_classes",
    "scaling_classes",
]
