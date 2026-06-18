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
from onescience.flax_models.alphagenome.model import one_hot_encoder
import numpy as np


class OneHotEncoderTest(parameterized.TestCase):

  @parameterized.product(
      sequence=['AGCTN', 'agctn'],
      output_type=[np.int8, np.float32],
  )
  def test_encode(self, sequence: str, output_type: np.dtype):
    expected = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=output_type,
    )
    encoder = one_hot_encoder.DNAOneHotEncoder(dtype=output_type)
    encoded = encoder.encode(sequence)
    np.testing.assert_array_equal(encoded, expected)


if __name__ == '__main__':
  absltest.main()
