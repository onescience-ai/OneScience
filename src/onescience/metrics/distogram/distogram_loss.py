"""Distogram loss implementation for AlphaFold3.

This module implements the distogram loss function that measures the difference
between predicted and true pairwise distance distributions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.utils.protenix.torch_utils import cdist


def loss_reduction(loss: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Applies reduction operation to loss tensor.

    Args:
        loss: Loss tensor of any shape [...].
        method: Reduction method. One of 'mean', 'sum', 'add', 'max', 'min', or None.
            Defaults to 'mean'.

    Returns:
        Reduced loss tensor. Shape: [] if method is not None, otherwise [...].
    """

    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes softmax cross entropy loss.

    Args:
        logits: Classification logits. Shape: [..., num_class].
        labels: Classification labels as probability distribution. Shape: [..., num_class].

    Returns:
        Softmax cross entropy loss. Shape: [...].
    """
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


class DistogramLoss(nn.Module):
    """Distogram loss for pairwise distance prediction.

    Implements the distogram loss from AlphaFold3. This loss is identical to AlphaFold2,
    where pairwise token distances use the representative atom for each token:
    - Cβ for protein residues (Cα for glycine)
    - C4' for purines and C2' for pyrimidines
    - Single atom per token for ligands
    """

    def __init__(
        self,
        min_bin: float = 2.3125,
        max_bin: float = 21.6875,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """Initializes the DistogramLoss module.

        Args:
            min_bin: Minimum boundary of distance bins in Angstroms. Defaults to 2.3125.
            max_bin: Maximum boundary of distance bins in Angstroms. Defaults to 21.6875.
            no_bins: Number of distance bins. Defaults to 64.
            eps: Small epsilon value added to denominator for numerical stability.
                Defaults to 1e-6.
            reduction: Loss reduction method ('mean', 'sum', or None). Defaults to 'mean'.
        """
        super(DistogramLoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
        self,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates distance bin labels from true coordinates.

        Args:
            true_coordinate: True atomic coordinates. Shape: [..., N_atom, 3].
            coordinate_mask: Mask indicating which coordinates exist.
                Shape: [N_atom] or [..., N_atom].
            rep_atom_mask: Mask for representative atoms. Shape: [N_atom].

        Returns:
            A tuple containing:
                - true_bins: One-hot encoded distance bins. Shape: [..., N_token, N_token, no_bins].
                - pair_coordinate_mask: Mask for valid representative atom pairs.
                    Shape: [N_token, N_token] or [..., N_token, N_token].
        """

        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins - 1,
            device=true_coordinate.device,
        )

        # Compute label: the true bins
        # True distance
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[..., rep_atom_mask, :]  # [..., N_token, 3]
        gt_dist = cdist(true_coordinate, true_coordinate)  # [..., N_token, N_token]
        # Assign distance to bins
        true_bins = torch.sum(
            gt_dist.unsqueeze(dim=-1) > boundaries, dim=-1
        )  # range in [0, no_bins-1], shape = [..., N_token, N_token]

        # Mask
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        return F.one_hot(true_bins, self.no_bins), pair_mask

    def forward(
        self,
        logits: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes distogram loss between predicted and true distances.

        Args:
            logits: Predicted distance distribution logits.
                Shape: [..., N_token, N_token, no_bins].
            true_coordinate: True atomic coordinates. Shape: [..., N_atom, 3].
            coordinate_mask: Mask indicating which coordinates exist.
                Shape: [N_atom] or [..., N_atom].
            rep_atom_mask: Mask for representative atoms. Shape: [N_atom].

        Returns:
            Distogram loss. Shape: [] if self.reduction is not None, otherwise [...].
        """

        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )

        errors = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., N_token, N_token]

        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = torch.sum(errors * pair_mask, dim=(-1, -2))
        loss = loss / denom

        return loss_reduction(loss, method=self.reduction)
