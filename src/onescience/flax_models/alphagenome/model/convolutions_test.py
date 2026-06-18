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

from absl.testing import absltest
from absl.testing import parameterized
from onescience.flax_models.alphagenome.model import convolutions
import chex
import haiku as hk
import jax
import jax.numpy as jnp


class ConvolutionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._batch_size = 4
    self._sequence_length = 200
    self._input_channels = 768
    self._output_channels = 256
    self._dna_embedder_output_channels = 768
    self._unet_skip_channels = 512
    self._down_block_added_channels = 128
    self._rng = jax.random.PRNGKey(42)

  @parameterized.named_parameters(
      dict(testcase_name='width_1', width=1),
      dict(testcase_name='width_5', width=5),
  )
  def test_conv_block_output_shape(self, width: int):
    def _conv_block(x):
      return convolutions.ConvBlock(
          num_channels=self._output_channels, width=width
      )(x)

    conv_block = hk.transform_with_state(_conv_block)
    x = jnp.zeros(
        (self._batch_size, self._sequence_length, self._input_channels)
    )
    params, state = conv_block.init(self._rng, x)
    out, state = conv_block.apply(params, state, self._rng, x)
    chex.assert_shape(
        out, (self._batch_size, self._sequence_length, self._output_channels)
    )
    self.assertNotEmpty(state)

  def test_standardized_conv1d_output_shape(self):
    def _standardized_conv1d(x):
      return convolutions.StandardizedConv1D(
          num_channels=self._output_channels, width=5
      )(x)

    standardized_conv1d = hk.transform_with_state(_standardized_conv1d)
    x = jnp.zeros(
        (self._batch_size, self._sequence_length, self._input_channels)
    )
    params, state = standardized_conv1d.init(self._rng, x)
    out, state = standardized_conv1d.apply(params, state, self._rng, x)
    chex.assert_shape(
        out, (self._batch_size, self._sequence_length, self._output_channels)
    )

  def test_dna_embedder_output_shape(self):
    def _dna_embedder(x):
      return convolutions.DnaEmbedder()(x)

    dna_embedder = hk.transform_with_state(_dna_embedder)
    x = jnp.zeros((self._batch_size, self._sequence_length, 4))
    params, state = dna_embedder.init(self._rng, x)
    out, state = dna_embedder.apply(params, state, self._rng, x)
    chex.assert_shape(
        out,
        (
            self._batch_size,
            self._sequence_length,
            self._dna_embedder_output_channels,
        ),
    )
    self.assertNotEmpty(state)

  def test_down_res_block_output_shape(self):
    def _down_res_block(x):
      return convolutions.DownResBlock()(x)

    down_res_block = hk.transform_with_state(_down_res_block)
    x = jnp.zeros(
        (self._batch_size, self._sequence_length, self._input_channels)
    )
    params, state = down_res_block.init(self._rng, x)
    out, state = down_res_block.apply(params, state, self._rng, x)
    chex.assert_shape(
        out,
        (
            self._batch_size,
            self._sequence_length,
            self._input_channels + self._down_block_added_channels,
        ),
    )
    self.assertNotEmpty(state)

  def test_up_res_block_output_shape(self):
    def _up_res_block(x, unet_skip):
      return convolutions.UpResBlock()(x, unet_skip)

    up_res_block = hk.transform_with_state(_up_res_block)
    x = jnp.zeros(
        (self._batch_size, self._sequence_length, self._input_channels)
    )
    unet_skip = jnp.zeros(
        (self._batch_size, self._sequence_length * 2, self._unet_skip_channels)
    )
    params, state = up_res_block.init(self._rng, x, unet_skip)
    out, state = up_res_block.apply(params, state, self._rng, x, unet_skip)
    chex.assert_shape(
        out,
        (
            self._batch_size,
            self._sequence_length * 2,
            self._unet_skip_channels,
        ),
    )
    self.assertNotEmpty(state)


if __name__ == '__main__':
  absltest.main()
