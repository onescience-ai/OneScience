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

"""Common layers."""

from onescience.flax_models.alphagenome._sdk import typing
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member, g-multiple-import


def gelu(x: jax.Array) -> jax.Array:
  """Gaussian Error Linear Unit activation function."""
  coef = jax.lax.convert_element_type(1.702, x.dtype)
  return jax.nn.sigmoid(coef * x) * x


@typing.jaxtyped
def pool(
    x: Float[Array, '... S D'], by: int = 2, reduce: str = 'max'
) -> Float[Array, '... S/{by} D']:
  """Applies pooling to the sequence dimension of the input.

  Args:
    x: The input sequence, where the second to last dimension is the sequence
      dimension.
    by: The pooling window size.
    reduce: The pooling reduction method.

  Returns:
    The pooled sequence.
  Raises:
    NotImplementedError: If the reduce method is not supported.
  """
  if reduce == 'max':
    return hk.MaxPool(window_shape=(by, 1), strides=(by, 1), padding='SAME')(x)
  elif reduce in ['avg', 'mean']:
    return hk.AvgPool(window_shape=(by, 1), strides=(by, 1), padding='SAME')(x)
  else:
    raise NotImplementedError(f'Reduce method={reduce} unknown.')


class RMSBatchNorm(hk.Module):
  r"""Root Mean Square Batch Normalization.

  Normalization is applied to the last dimension of the input as
  `x -> x * scale / sqrt(var + epsilon) + offset`.
  The scale and offset are learned parameters. The variance is tracked
  as an exponential moving average.

  Variance is computed across the batch and sequence dimension.

  Note: only support inference mode (i.e. no training).
  """

  def __call__(self, x: Float[Array, '... D']) -> Float[Array, '... D']:
    param_shape = (1,) * (x.ndim - 1) + (x.shape[-1],)
    variance = hk.get_state(
        'var_ema', param_shape, dtype=jnp.float32, init=jnp.ones
    )
    scale = hk.get_parameter(
        'scale', param_shape, dtype=x.dtype, init=jnp.ones
    ).astype(x.dtype)
    offset = hk.get_parameter(
        'offset', param_shape, dtype=x.dtype, init=jnp.zeros
    )
    inv = scale * jax.lax.rsqrt(variance + 1e-5).astype(x.dtype)
    return x * inv + offset


class LayerNorm(hk.Module):
  """Layer Normalization."""

  def __init__(
      self, rms_norm: bool = False, axis: int = -1, name: str | None = None
  ) -> None:
    """Initializes the LayerNorm module.

    Args:
      rms_norm: If False, the input is centered before computing the
        mean-squared for normalization. If True, the mean-squared is computed
        directly on the uncentered input.
      axis: The axis to apply the normalization to.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._rms_norm = rms_norm
    self._axis = axis

  def __call__(self, x: Float[Array, '... D']) -> Float[Array, '... D']:
    dtype = x.dtype
    scale = hk.get_parameter(
        'scale', (x.shape[-1],), dtype, init=jnp.ones
    ).astype(dtype)
    offset = hk.get_parameter(
        'offset', (x.shape[-1],), dtype, init=jnp.zeros
    ).astype(dtype)
    scale = jax.lax.broadcast_to_rank(scale, x.ndim)
    offset = jax.lax.broadcast_to_rank(offset, x.ndim)

    if not self._rms_norm:
      mean = jnp.mean(
          x, axis=self._axis, dtype=jnp.float32, keepdims=True
      ).astype(dtype)
      x = x - mean

    variance = jnp.mean(
        jnp.square(x), axis=self._axis, dtype=jnp.float32, keepdims=True
    )
    inv = scale * jax.lax.rsqrt(variance + 1e-5).astype(dtype)
    return inv * x + offset
