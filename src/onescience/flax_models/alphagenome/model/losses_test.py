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
from onescience.flax_models.alphagenome.model import losses
import jax.numpy as jnp
import numpy as np


class LossesTest(parameterized.TestCase):

  def test_multinomial_loss_masking(self):
    """Tests that masking correctly zeros out predictions and targets."""
    y_true = jnp.array([[[10.0, 1.0, 3.0], [5.0, 2.0, 20.0]]])
    y_pred = jnp.array([[[0.5, 2.5, 1.0], [2.5, 0.5, 1.0]]])
    loss_full = losses.multinomial_loss(
        y_true=y_true,
        y_pred=y_pred,
        mask=jnp.array([[[True, True, True]]]),
        multinomial_resolution=1,
        positional_weight=1.0,
    )['loss']
    loss_masked = losses.multinomial_loss(
        y_true=y_true,
        y_pred=y_pred,
        mask=jnp.array([[[True, True, False]]]),
        multinomial_resolution=1,
        positional_weight=1.0,
    )['loss']
    y_true_zero = jnp.array([[[10.0, 1.0], [5.0, 2.0]]])
    y_pred_zero = jnp.array([[[0.5, 2.5], [2.5, 0.5]]])
    loss_truncated = losses.multinomial_loss(
        y_true=y_true_zero,
        y_pred=y_pred_zero,
        mask=jnp.array([[[True, True]]]),
        multinomial_resolution=1,
        positional_weight=1.0,
    )['loss']
    np.testing.assert_almost_equal(loss_masked, loss_truncated, decimal=5)
    np.testing.assert_array_less(loss_masked, loss_full)

  def test_multinomial_loss_resolution_aggregation(self):
    """Tests the resolution aggregation logic."""
    # Seq length 4, all 1s.
    y_true = jnp.ones((1, 4, 1))
    y_pred = jnp.ones((1, 4, 1))
    mask = jnp.ones((1, 1, 1), dtype=bool)
    out_res1 = losses.multinomial_loss(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        multinomial_resolution=1,
        positional_weight=1.0,
    )
    out_res4 = losses.multinomial_loss(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        multinomial_resolution=4,
        positional_weight=1.0,
    )
    self.assertTrue(np.all(np.isfinite(out_res1['loss'])))
    self.assertTrue(np.all(np.isfinite(out_res4['loss'])))
    np.testing.assert_almost_equal(out_res1['max_sum_preds'], 1.0)
    np.testing.assert_almost_equal(out_res4['max_sum_preds'], 4.0)


if __name__ == '__main__':
  absltest.main()
