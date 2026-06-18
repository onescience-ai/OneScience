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
from onescience.flax_models.alphagenome.model import splicing
import jax.numpy as jnp
import jaxtyping
import numpy as np


class SplicingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Disable jaxtyped decorator for testing as parameterized
    # test_generate_splice_site_positions test doesn't follow the jaxtyped
    # annotations.
    jaxtyping.config.update('jaxtyping_disable', True)

  def test_generate_splice_site_positions_no_alt_no_mask(self):
    # ref probabilities not normalized but it doesn't matter for this test.
    ref = jnp.array([[
        [0.09, 0.2, 0.9, 0.4, 0.0],
        [0.5, 0.6, 0.7, 0.05, 0.0],
        [0.05, 0.8, 0.6, 0.6, 0.0],
    ]])
    k = 2
    pad_to_length = 5
    threshold = 0.1

    result = splicing.generate_splice_site_positions(
        ref,
        alt=None,
        splice_sites=None,
        k=k,
        pad_to_length=pad_to_length,
        threshold=threshold,
    )
    expected = jnp.array([[
        [1, -1, -1, -1, -1],  # donor +ve
        [1, 2, -1, -1, -1],  # acceptor +ve
        [0, 1, -1, -1, -1],  # donor -ve
        [0, 2, -1, -1, -1],  # acceptor -ve
    ]])
    np.testing.assert_array_equal(result, expected)

  @parameterized.parameters(
      dict(
          ref=np.array([[[0.1], [0.2], [0.9], [0.8], [0.0]]]),
          alt=None,
          splice_site_mask=None,
          probability_threshold=0.0,
          expected=np.array([[[2, 3]]]),
      ),
      dict(
          ref=np.array([[[0.1], [0.2], [0.9], [0.8], [0.0]]]),
          alt=np.array([[[0.9], [0.1], [0.1], [0.1], [0.0]]]),
          splice_site_mask=None,
          probability_threshold=0.0,
          expected=np.array([[[0, 2]]]),
      ),
      dict(
          ref=np.array([[[0.0], [0.2], [0.9], [0.8], [0.0]]]),
          alt=None,
          splice_site_mask=np.array([[[0.05], [0.1], [0.1], [0.1], [0.9]]]),
          probability_threshold=0.1,
          expected=np.array([[[2, 4]]]),
      ),
      dict(
          ref=np.array([[[0.1], [0.2], [0.9], [0.8], [0.0]]]),
          alt=None,
          splice_site_mask=None,
          probability_threshold=0.5,
          expected=np.array([[[2, 3]]]),
      ),
      dict(
          ref=np.array([[[0.1], [0.2], [0.3], [0.4], [0.0]]]),
          alt=None,
          splice_site_mask=None,
          probability_threshold=0.5,
          expected=np.array([[[-1, -1]]]),
      ),
  )
  def test_generate_splice_site_positions(
      self, ref, alt, splice_site_mask, probability_threshold, expected
  ):
    actual = splicing.generate_splice_site_positions(
        jnp.array(ref),
        alt=jnp.array(alt) if alt is not None else None,
        splice_sites=jnp.array(splice_site_mask)
        if splice_site_mask is not None
        else None,
        k=2,
        pad_to_length=1,
        threshold=probability_threshold,
    )
    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()
