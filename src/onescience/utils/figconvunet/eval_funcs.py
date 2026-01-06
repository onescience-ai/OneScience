from typing import Dict, Optional
from jaxtyping import Float

import torch
from torch import Tensor


def r2_score(
    y_true: Float[Tensor, "B"], y_pred: Float[Tensor, "B"]
) -> Float[Tensor, "1"]:
    """Compute the R^2 score."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    return 1 - ss_res / ss_tot


def mean_squared_error(
    y_true: Float[Tensor, "B"], y_pred: Float[Tensor, "B"]
) -> Float[Tensor, "1"]:
    """Compute the mean squared error."""
    return ((y_true - y_pred) ** 2).mean()


def mean_absolute_error(
    y_true: Float[Tensor, "B"], y_pred: Float[Tensor, "B"]
) -> Float[Tensor, "1"]:
    """Compute the mean absolute error."""
    return (y_true - y_pred).abs().mean()


def max_absolute_error(
    y_true: Float[Tensor, "B"], y_pred: Float[Tensor, "B"]
) -> Float[Tensor, "1"]:
    """Compute the maximum absolute error."""
    return (y_true - y_pred).abs().max()


def rrmse(y_true: Float[Tensor, "B"], y_pred: Float[Tensor, "B"]) -> Float[Tensor, "1"]:
    """Compute the relative RMSE."""
    return torch.linalg.vector_norm(y_pred - y_true) / torch.linalg.vector_norm(y_true)


def eval_all_metrics(
    y_true: Float[Tensor, "B"], y_pred: Float[Tensor, "B"], prefix: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate all metrics."""
    # detach to avoid memory leak
    y_true = y_true.detach()
    y_pred = y_pred.detach()

    # Assert that the shapes are the same
    assert y_true.shape == y_pred.shape

    out_dict = {
        "r2": r2_score(y_true, y_pred).cpu().item(),
        "mse": mean_squared_error(y_true, y_pred).cpu().item(),
        "mae": mean_absolute_error(y_true, y_pred).cpu().item(),
        "maxae": max_absolute_error(y_true, y_pred).cpu().item(),
        "rrmse": rrmse(y_true, y_pred).cpu().item(),
    }
    if prefix is not None:
        out_dict = {f"{prefix}_{k}": v for k, v in out_dict.items()}
    return out_dict
