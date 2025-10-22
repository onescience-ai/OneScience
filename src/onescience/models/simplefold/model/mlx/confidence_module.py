#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import mlx.nn as nn
import mlx.core as mx


def compute_aggregated_metric(logits, end=1.0):
    """Compute the metric from the logits.

    Parameters
    ----------
    logits : torch.Tensor
        The logits of the metric
    end : float
        Max value of the metric, by default 1.0

    Returns
    -------
    Tensor
        The metric value

    """
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = mx.arange(start=0.5 * bin_width, stop=end, step=bin_width)
    probs = mx.softmax(logits, axis=-1)
    plddt = mx.sum(
        probs * bounds.reshape(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        axis=-1,
    )
    return plddt


class ConfidenceModule(nn.Module):
    def __init__(
        self,
        hidden_size,
        transformer_blocks=None,
        num_plddt_bins=50,
    ):
        super().__init__()
        self.transformer_blocks = transformer_blocks
        self.to_plddt_logits = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_plddt_bins),
        )

    def __call__(
        self,
        latent,
        feats,
    ):
        if self.transformer_blocks is not None:
            token_pe_pos = mx.concatenate(
                [
                    feats["residue_index"][..., None].astype(mx.float32),  # (B, M, 1)
                    feats["entity_id"][..., None].astype(mx.float32),  # (B, M, 1)
                    feats["asym_id"][..., None].astype(mx.float32),  # (B, M, 1)
                    feats["sym_id"][..., None].astype(mx.float32),  # (B, M, 1)
                ],
                axis=-1,
            )

            latent = self.transformer_blocks(
                latents=latent,
                c=None,
                pos=token_pe_pos,
            )

        # Compute the pLDDT
        plddt_logits = self.to_plddt_logits(latent)

        # Compute the aggregated pLDDT
        plddt = compute_aggregated_metric(plddt_logits)

        return dict(
            plddt=plddt,
            plddt_logits=plddt_logits,
        )
