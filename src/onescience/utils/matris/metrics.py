"""
Metrics for MatRIS model evaluation.
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Metrics:
    """Container for metrics."""
    mae: float = 0.0
    rmse: float = 0.0
    count: int = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update metrics with new predictions and targets."""
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        diff = pred - target
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        
        # Weighted average
        total = self.count + len(pred)
        self.mae = (self.mae * self.count + mae * len(pred)) / total
        self.rmse = (self.rmse * self.count + rmse * len(pred)) / total
        self.count = total
    
    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to dictionary."""
        return {
            f"{prefix}mae": self.mae,
            f"{prefix}rmse": self.rmse,
        }


def compute_metrics(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    task: str = "ef",
) -> dict[str, Metrics]:
    """
    Compute metrics for predictions.
    
    Args:
        predictions: Dict with predictions
        targets: Dict with targets
        task: Task type
        
    Returns:
        Dict of Metrics objects
    """
    metrics = {}
    
    # Energy metrics
    if "e" in task and "e" in predictions and "e" in targets:
        energy_metrics = Metrics()
        energy_metrics.update(predictions["e"], targets["e"])
        metrics["energy"] = energy_metrics
    
    # Force metrics
    if "f" in task and "f" in predictions and "f" in targets:
        force_metrics = Metrics()
        # Flatten forces
        pred_forces = predictions["f"].reshape(-1, 3)
        target_forces = targets["f"].reshape(-1, 3)
        force_metrics.update(pred_forces, target_forces)
        metrics["force"] = force_metrics
    
    # Stress metrics
    if "s" in task and "s" in predictions and "s" in targets:
        stress_metrics = Metrics()
        pred_stress = predictions["s"].reshape(-1, 9)
        target_stress = targets["s"].reshape(-1, 9)
        stress_metrics.update(pred_stress, target_stress)
        metrics["stress"] = stress_metrics
    
    # Magnetic moment metrics
    if "m" in task and "m" in predictions and "m" in targets:
        magmom_metrics = Metrics()
        magmom_metrics.update(predictions["m"], targets["m"])
        metrics["magmom"] = magmom_metrics
    
    return metrics
