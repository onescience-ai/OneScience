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

"""Embeddings for AlphaGenome."""

from onescience.flax_models.alphagenome._sdk import typing
from onescience.flax_models.alphagenome.model import layers
import chex
import haiku as hk
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-importing-member, g-multiple-import


@typing.jaxtyped
@chex.dataclass(frozen=True, kw_only=True)
class Embeddings:
  """AlphaGenome embeddings."""

  embeddings_1bp: Float[Array, 'B S 1536'] | None = None
  embeddings_128bp: Float[Array, 'B S//128 3072'] | None = None
  embeddings_pair: Float[Array, 'B S//2048 S//2048 128'] | None = None

  def get_sequence_embeddings(self, resolution: int) -> Float[Array, 'B S D']:
    if resolution == 128:
      return self.embeddings_128bp
    elif resolution == 1:
      return self.embeddings_1bp
    else:
      raise ValueError(f'Unsupported resolution: {resolution}')


class OutputEmbedder(hk.Module):
  """Generates output embeddings with organism-specific adjustments."""

  def __init__(self, num_organisms: int, name: str | None = None):
    """Initializes the OutputEmbedder module.

    Args:
      num_organisms: The number of organisms to embed. Typically 2 (human and
        mouse).
      name: The name of the module.
    """
    super().__init__(name=name)
    self._num_organisms = num_organisms

  @typing.jaxtyped
  def __call__(
      self,
      x: Float[Array, 'B S D'],
      organism_index: Int[Array, 'B'],
      skip_x: Float[Array, 'B S_skip D_skip'] | None = None,
  ) -> Float[Array, 'B S D_out']:
    x = hk.Linear(2 * x.shape[-1])(x)
    if skip_x is not None:
      # Assumes skip_x needs to be upsampled to match x's sequence length
      skip_x = hk.Linear(x.shape[-1], with_bias=False)(skip_x)
      x += jnp.repeat(skip_x, x.shape[1] // skip_x.shape[1], axis=1)

    x = layers.RMSBatchNorm()(x)
    if self._num_organisms >= 1:
      organism_embedding = hk.Embed(self._num_organisms, x.shape[-1])(
          organism_index
      )[:, None, :]
      x += organism_embedding
    return layers.gelu(x)


class OutputPair(hk.Module):
  """Generates output pairwise embeddings with organism-specific adjustments."""

  def __init__(self, num_organisms: int, name: str | None = None):
    """Initializes the OutputPair module.

    Args:
      num_organisms: The number of organisms to embed. Typically 2 (human and
        mouse).
      name: The name of the module.
    """
    super().__init__(name=name)
    self._num_organisms = num_organisms

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S S F'], organism_index: Int[Array, 'B']
  ) -> Float[Array, 'B S S 128']:
    x = (x + jnp.swapaxes(x, 1, 2)) / 2.0  # Symmetrize.
    x = layers.LayerNorm(rms_norm=True)(x)
    if self._num_organisms >= 1:
      organism_embedding = hk.Embed(self._num_organisms, 128)(organism_index)
      x += organism_embedding[:, None, None, :]
    return layers.gelu(x)
