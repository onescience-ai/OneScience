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
from onescience.flax_models.alphagenome.model import layers
import chex
import haiku as hk
import jax.numpy as jnp


def _get_pool_layer(by: int):
  """Helper function to create the pool layer."""

  def pool_fn(x):
    return layers.pool(x, by=by)

  pool_mod = hk.transform(pool_fn)
  return pool_mod


class LayersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._batch_size = 4
    self._sequence_length = 16
    self._num_channels = 8
    self._by = 2

  def test_pool_shape(self):
    """Tests that the pool layer returns the expected shape."""
    x = jnp.ones((self._batch_size, self._sequence_length, self._num_channels))
    pool_mod = _get_pool_layer(by=self._by)
    params = pool_mod.init(None, x)
    out = pool_mod.apply(params, None, x)
    chex.assert_shape(
        out,
        (
            self._batch_size,
            self._sequence_length // self._by,
            self._num_channels,
        ),
    )

  def test_pool_type_error(self):
    """Tests that the pool layer raises a type error with invalid input."""
    with self.subTest('wrong_dtype'):
      x = jnp.ones(
          (self._batch_size, self._sequence_length, self._num_channels),
          dtype=jnp.int32,  # Provide wrong type.
      )
      with self.assertRaises(TypeError):
        pool_mod = _get_pool_layer(by=self._by)
        params = pool_mod.init(None, x)
        pool_mod.apply(params, None, x)


if __name__ == '__main__':
  absltest.main()
