# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Splicing utilities."""

from onescience.flax_models.alphagenome._sdk import typing
from onescience.flax_models.alphagenome.model.variant_scoring import splice_junction
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int32  # pylint: disable=g-importing-member, g-multiple-import


@typing.jaxtyped
def _top_k_splice_sites(
    x: Float[Array, 'B S 5'],
    *,
    k: int,
    pad_to_length: int,
    threshold: float,
) -> Int32[Array, 'B 4 K']:
  """Returns the top k splice sites from the predictions.

  Args:
    x: Array of shape [batch, sequence_length, 5] containing splice site
      predictions (donor +ve, acceptor +ve, donor -ve, acceptor -ve, other).
    k: Number of top splice sites to return.
    pad_to_length: Pad the output to this length.
    threshold: Threshold to filter out low confidence splice sites.
  """
  batch_size = x.shape[0]
  values, positions = jax.lax.approx_max_k(x[..., :4], k, reduction_dimension=1)
  if threshold > 0:
    positions = jnp.where(values < threshold, jnp.inf, positions)
  positions = jnp.sort(positions, axis=1, descending=False)
  if threshold > 0:
    positions = jnp.where(positions == jnp.inf, -1, positions)
  # positions shape [batch, 4 (+ve and -ve donors and acceptors), k].
  positions = positions.swapaxes(1, 2).astype(jnp.int32)
  if positions.shape[2] < pad_to_length:
    padding_shape = (batch_size, 4, pad_to_length - positions.shape[2])
    padding = jnp.full(
        padding_shape, splice_junction.PAD_VALUE, dtype=jnp.int32
    )
    positions = jnp.concatenate([positions, padding], axis=2)
  return positions


@typing.jaxtyped
def generate_splice_site_positions(
    ref: Float[Array, 'B S 5'],
    alt: Float[Array, 'B S 5'] | None,
    splice_sites: Bool[Array, 'B S 5'] | None,
    *,
    k: int,
    pad_to_length: int,
    threshold: float,
) -> Int32[Array, 'B 4 K']:
  """Returns the top k splice sites from predictions and (true) splice sites."""

  if alt is not None:
    ref = jnp.maximum(ref, alt)
  if splice_sites is not None:
    ref = jnp.maximum(ref, splice_sites)
  return _top_k_splice_sites(
      ref, k=k, pad_to_length=pad_to_length, threshold=threshold
  )
