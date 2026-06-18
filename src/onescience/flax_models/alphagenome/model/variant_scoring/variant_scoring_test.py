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
from onescience.flax_models.alphagenome._sdk.data import genome
from onescience.flax_models.alphagenome.model.variant_scoring import variant_scoring
import chex
import numpy as np


class AlignAlternateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='snp',
          variant=genome.Variant('chr1', 2, 'C', 'T'),
          expected_aligned_alt=np.arange(10, dtype=np.float32).reshape(-1, 1),
      ),
      dict(
          testcase_name='insertion',
          variant=genome.Variant('chr1', 2, 'C', 'TGA'),
          expected_aligned_alt=np.array(
              [0, 3, 4, 5, 6, 7, 8, 9, 0, 0], dtype=np.float32
          ).reshape(-1, 1),
      ),
      dict(
          testcase_name='multi_insertion',
          variant=genome.Variant('chr1', 2, 'CC', 'CCTGA'),
          expected_aligned_alt=np.array(
              [0, 1, 5, 6, 7, 8, 9, 0, 0, 0], dtype=np.float32
          ).reshape(-1, 1),
      ),
      dict(
          testcase_name='multi_insertion_long_insertion',
          variant=genome.Variant('chr1', 5, 'A', 'AAAAAAA'),
          expected_aligned_alt=np.array(
              [0, 1, 2, 3, 9, 0, 0, 0, 0, 0], dtype=np.float32
          ).reshape(-1, 1),
      ),
      dict(
          testcase_name='deletion',
          variant=genome.Variant('chr1', 2, 'CGA', 'C'),
          expected_aligned_alt=np.array(
              [0, 1, 0, 0, 2, 3, 4, 5, 6, 7], dtype=np.float32
          ).reshape(-1, 1),
      ),
      dict(
          testcase_name='multi_deletion',
          variant=genome.Variant('chr1', 2, 'CCCGA', 'CCC'),
          expected_aligned_alt=np.array(
              [0, 1, 2, 3, 0, 0, 4, 5, 6, 7], dtype=np.float32
          ).reshape(-1, 1),
      ),
  )
  def test_align_alt(self, variant, expected_aligned_alt):
    interval = genome.Interval('chr1', 0, 10)
    alt = np.arange(10, dtype=np.float32).reshape(-1, 1)
    aligned_alt = variant_scoring.align_alternate(alt, variant, interval)
    chex.assert_shape(aligned_alt, (10, 1))
    np.testing.assert_array_equal(aligned_alt, expected_aligned_alt)


if __name__ == '__main__':
  absltest.main()
