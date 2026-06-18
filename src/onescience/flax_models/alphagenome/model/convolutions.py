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

"""Convolutional layers."""

from onescience.flax_models.alphagenome._sdk import typing
from onescience.flax_models.alphagenome.model import layers
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member, g-multiple-import


class ConvBlock(hk.Module):
  """A convolutional block with GELU activation and RMS Batch Normalization."""

  def __init__(self, num_channels: int, width: int, name: str | None = None):
    """Initializes the convolutional block.

    Args:
      num_channels: The number of output channels.
      width: The width of the convolution. If 1, a linear layer is used instead.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._num_channels = num_channels
    self._width = width

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D']
  ) -> Float[Array, 'B S {self._num_channels}']:
    x = layers.gelu(layers.RMSBatchNorm()(x))
    if self._width == 1:
      return hk.Linear(self._num_channels)(x)
    else:
      return StandardizedConv1D(
          num_channels=self._num_channels, width=self._width
      )(x)


class StandardizedConv1D(hk.Module):
  """Standardized 1D Convolution with scaled weight standardization."""

  def __init__(
      self,
      num_channels: int,
      width: int,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_channels = num_channels
    self._width = width

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D']
  ) -> Float[Array, 'B S {self._num_channels}']:
    input_channels = x.shape[-1]
    fan_in = self._width * input_channels
    kernel_shape = (self._width, input_channels, self._num_channels)
    w = hk.get_parameter('w', shape=kernel_shape, dtype=x.dtype, init=jnp.zeros)

    # Weight standardization
    w -= jnp.mean(w, axis=(0, 1), keepdims=True)
    var_w = jnp.var(w, axis=(0, 1), keepdims=True)
    scale = hk.get_parameter(
        'scale',
        shape=[1, 1, self._num_channels],
        init=jnp.ones,
        dtype=w.dtype,
    )
    scale = scale * jax.lax.rsqrt(jnp.maximum(fan_in * var_w, 1e-4))
    w_standardized = w * scale

    out = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w_standardized,
        window_strides=[1],
        padding='SAME',
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 2, 1), rhs_spec=(2, 1, 0), out_spec=(0, 2, 1)
        ),
    )
    bias = hk.get_parameter(
        'bias', shape=(self._num_channels,), dtype=x.dtype, init=jnp.zeros
    )
    bias = jnp.broadcast_to(bias, out.shape)
    return out + bias


class DnaEmbedder(hk.Module):
  """Encodes one-hot encoded DNA sequences into a fixed-length embedding."""

  @typing.jaxtyped
  def __call__(
      self, dna_sequence: Float[Array, 'B S 4']
  ) -> Float[Array, 'B S 768']:
    x = hk.Conv1D(output_channels=768, kernel_shape=15)(dna_sequence)
    return x + ConvBlock(num_channels=768, width=5)(x)


class DownResBlock(hk.Module):
  """Down resolution convolution."""

  @typing.jaxtyped
  def __call__(self, x: Float[Array, 'B S D']) -> Float[Array, 'B S D+128']:
    num_out_channels = x.shape[-1] + 128
    out = ConvBlock(num_channels=num_out_channels, width=5)(x)
    out = out + jnp.pad(
        x, [(0, 0), (0, 0), (0, 128)]
    )  # Padding for residual connection
    return out + ConvBlock(num_channels=out.shape[-1], width=5)(out)


class UpResBlock(hk.Module):
  """Upsampling residual block."""

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D'], unet_skip: Float[Array, 'B S_skip D_skip']
  ) -> Float[Array, 'B S_up D_skip']:
    chex.assert_rank(x, 3)
    chex.assert_rank(unet_skip, 3)
    num_channels = unet_skip.shape[-1]
    out = (
        ConvBlock(num_channels=num_channels, width=5, name='conv_in')(x)
        + x[:, :, :num_channels]
    )
    out = jnp.repeat(out, 2, axis=1)  # Upsampling
    residual_scale = hk.get_parameter(
        'residual_scale', (), init=jnp.ones
    ).astype(out.dtype)
    out *= residual_scale
    out += ConvBlock(
        num_channels=num_channels, width=1, name='pointwise_conv_unet_skip'
    )(unet_skip)
    return out + ConvBlock(num_channels=num_channels, width=5, name='conv_out')(
        out
    )
