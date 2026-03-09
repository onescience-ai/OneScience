"""Distogram head module for AlphaFold3.

This module implements the distogram prediction head that computes pairwise distance
distributions between tokens, as described in Algorithm 1 (Line 17) of AlphaFold3.
Adapted from OpenFold model heads.
"""

import torch
import torch.nn as nn
from onescience.modules.linear.protenixlinear import ProtenixLinear


class DistogramHead(nn.Module):
    """Computes distogram probability distributions for token pairs.

    Implements Algorithm 1 [Line 17] in AlphaFold3. The distogram represents
    pairwise distance distributions and is used for computing distogram loss
    as described in subsection 1.9.8 of AlphaFold2.
    """

    def __init__(self, c_z: int = 128, no_bins: int = 64) -> None:
        """Initializes the DistogramHead module.

        Args:
            c_z: Hidden dimension for pair embeddings. Defaults to 128.
            no_bins: Number of distance bins for the distogram. Defaults to 64.
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = ProtenixLinear(
            in_features=self.c_z, out_features=self.no_bins, initializer="zeros"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Computes distogram logits from pair embeddings.

        Args:
            z: Pair embeddings. Shape: [*, N_token, N_token, c_z].

        Returns:
            Symmetrized distogram logits. Shape: [*, N_token, N_token, no_bins].
        """
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
