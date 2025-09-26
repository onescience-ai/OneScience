"""Distogram head."""

from typing import Final

import haiku as hk
import jax
import jax.numpy as jnp

from onescience.flax_models.alphafold3.common import base_config
from onescience.flax_models.alphafold3.model import feat_batch, model_config
from onescience.flax_models.alphafold3.model.components import haiku_modules as hm

_CONTACT_THRESHOLD: Final[float] = 8.0
_CONTACT_EPSILON: Final[float] = 1e-3


class DistogramHead(hk.Module):
    """Distogram head."""

    class Config(base_config.BaseConfig):
        first_break: float = 2.3125
        last_break: float = 21.6875
        num_bins: int = 64

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        name="distogram_head",
    ):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        pair_act = embeddings["pair"]
        seq_mask = batch.token_features.mask.astype(bool)
        pair_mask = seq_mask[:, None] * seq_mask[None, :]

        left_half_logits = hm.Linear(
            self.config.num_bins,
            initializer=self.global_config.final_init,
            name="half_logits",
        )(pair_act)

        right_half_logits = left_half_logits
        logits = left_half_logits + \
            jnp.swapaxes(right_half_logits, -2, -3)
        probs = jax.nn.softmax(logits, axis=-1)
        breaks = jnp.linspace(
            self.config.first_break,
            self.config.last_break,
            self.config.num_bins - 1,
        )

        bin_tops = jnp.append(
            breaks, breaks[-1] + (breaks[-1] - breaks[-2]))
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)
        contact_probs = jnp.einsum(
            "ijk,k->ij", probs, is_contact_bin, precision=jax.lax.Precision.HIGHEST
        )
        contact_probs = pair_mask * contact_probs

        return {
            "bin_edges": breaks,
            "contact_probs": contact_probs,
        }
