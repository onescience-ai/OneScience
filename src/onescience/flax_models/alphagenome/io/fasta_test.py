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

import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
from onescience.flax_models.alphagenome._sdk.data import genome
from onescience.flax_models.alphagenome.io import fasta


def _get_test_fasta_path() -> str | os.PathLike[str]:
  return pathlib.Path(__file__).parent / 'testdata' / 'example.fa'


class FastaExtractorTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(interval='chr1:0-9:+', expected='TAATATGCT'),
      dict(interval='chr1:0-9:-', expected='AGCATATTA'),
      dict(interval='chr2:70-78', expected='CAGATACA'),
      dict(interval='chr1:-10-1', expected='NNNNNNNNNNT'),
      dict(interval='chr1:50-55:.', expected='ANNNN'),
      dict(interval='chr1:50-55:-', expected='NNNNT'),
  ])
  def test_extract(self, interval: str, expected: str):
    extractor = fasta.FastaExtractor(_get_test_fasta_path())
    interval = genome.Interval.from_str(interval)
    self.assertEqual(extractor.extract(interval), expected)

  def test_extract_fully_out_of_bounds_raises(self):
    extractor = fasta.FastaExtractor(_get_test_fasta_path())
    with self.assertRaisesRegex(ValueError, 'Interval fully out of bounds.'):
      extractor.extract(genome.Interval('chr1', 100, 1000))

  def test_invalid_chromosome_raises(self):
    extractor = fasta.FastaExtractor(_get_test_fasta_path())
    with self.assertRaisesRegex(ValueError, 'Chromosome "foo" not found.'):
      extractor.extract(genome.Interval('foo', 0, 10))


if __name__ == '__main__':
  absltest.main()
