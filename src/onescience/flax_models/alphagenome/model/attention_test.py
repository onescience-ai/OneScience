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
from onescience.flax_models.alphagenome.model import attention
import chex
import haiku as hk
import jax
import jax.numpy as jnp


class ConvolutionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._batch_size = 4
    self._sequence_length = 4096
    self._pair_sequence_length = self._sequence_length // 16
    self._hidden_size = 64

  def test_mlp_block_output_shape(self):
    """Tests that MLPBlock produces the expected output shape."""
    rng = jax.random.PRNGKey(42)

    def _forward(x):
      module = attention.MLPBlock()
      return module(x)

    init, apply = hk.transform_with_state(_forward)
    x = jnp.zeros((self._batch_size, self._sequence_length, self._hidden_size))
    params, state = init(rng, x)
    output, _ = apply(params, state, rng, x)
    chex.assert_shape(
        output, (self._batch_size, self._sequence_length, self._hidden_size)
    )

  def test_pair_mlp_block_output_shape(self):
    """Tests that PairMLPBlock produces the expected output shape."""
    rng = jax.random.PRNGKey(42)

    def _forward(x):
      module = attention.PairMLPBlock()
      return module(x)

    init, apply = hk.transform_with_state(_forward)
    x = jnp.zeros((
        self._batch_size,
        self._pair_sequence_length,
        self._pair_sequence_length,
        self._hidden_size,
    ))
    params, state = init(rng, x)
    output, _ = apply(params, state, rng, x)
    chex.assert_shape(
        output,
        (
            self._batch_size,
            self._pair_sequence_length,
            self._pair_sequence_length,
            self._hidden_size,
        ),
    )


if __name__ == "__main__":
  absltest.main()
