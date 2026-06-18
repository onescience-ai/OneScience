"""
Loss functions for MatRIS model.
"""
from __future__ import annotations

import torch
from torch import nn


class MatrisLoss(nn.Module):
    """
    Combined loss for MatRIS model supporting energy, forces, stress, and magnetic moments.
    """
    
    def __init__(
        self,
        energy_weight: float = 1.0,
        force_weight: float = 10.0,
        stress_weight: float = 0.1,
        magmom_weight: float = 0.1,
        task: str = "ef",
    ):
        """
        Args:
            energy_weight: Weight for energy loss
            force_weight: Weight for force loss
            stress_weight: Weight for stress loss
            magmom_weight: Weight for magnetic moment loss
            task: Task type ('e', 'ef', 'efs', 'efsm', 'em')
        """
        super().__init__()
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.magmom_weight = magmom_weight
        self.task = task
        
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        num_atoms: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            predictions: Dict with keys 'e', 'f', 's', 'm' depending on task
            targets: Dict with same keys as predictions
            num_atoms: Number of atoms per graph (for normalization)
            
        Returns:
            Dict with total loss and individual losses
        """
        losses = {}
        total_loss = 0.0
        
        # Energy loss
        if "e" in self.task and "e" in predictions and "e" in targets:
            energy_loss = self.mse(predictions["e"], targets["e"])
            losses["energy"] = energy_loss
            total_loss += self.energy_weight * energy_loss
        
        # Force loss
        if "f" in self.task and "f" in predictions and "f" in targets:
            force_loss = self.mse(predictions["f"], targets["f"])
            losses["force"] = force_loss
            total_loss += self.force_weight * force_loss
        
        # Stress loss
        if "s" in self.task and "s" in predictions and "s" in targets:
            # Stress might need special handling for shape
            pred_stress = predictions["s"]
            target_stress = targets["s"]
            if pred_stress.dim() == 3 and target_stress.dim() == 2:
                target_stress = target_stress.unsqueeze(0)
            stress_loss = self.mse(pred_stress, target_stress)
            losses["stress"] = stress_loss
            total_loss += self.stress_weight * stress_loss
        
        # Magnetic moment loss
        if "m" in self.task and "m" in predictions and "m" in targets:
            magmom_loss = self.mse(predictions["m"], targets["m"])
            losses["magmom"] = magmom_loss
            total_loss += self.magmom_weight * magmom_loss
        
        losses["total"] = total_loss
        return losses
