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

"""A one-hot encoder for DNA sequences."""

import numpy as np
import numpy.typing as np_typing


class DNAOneHotEncoder:
  """A one-hot encoder for DNA sequences.

  A -> [1, 0, 0, 0]
  C -> [0, 1, 0, 0]
  G -> [0, 0, 1, 0]
  T -> [0, 0, 0, 1]

  all other characters are encoded as zeros [0, 0, 0, 0].
  """

  def __init__(self, dtype: np_typing.DTypeLike = np.float32):
    self._lookup_table = np.zeros((256, 4), dtype=dtype)
    self._lookup_table[ord('A')] = [1, 0, 0, 0]
    self._lookup_table[ord('C')] = [0, 1, 0, 0]
    self._lookup_table[ord('G')] = [0, 0, 1, 0]
    self._lookup_table[ord('T')] = [0, 0, 0, 1]
    self._lookup_table[ord('a')] = self._lookup_table[ord('A')]
    self._lookup_table[ord('c')] = self._lookup_table[ord('C')]
    self._lookup_table[ord('g')] = self._lookup_table[ord('G')]
    self._lookup_table[ord('t')] = self._lookup_table[ord('T')]

  def encode(self, seq: str) -> np.ndarray:
    """One-hot encodes a DNA sequence string.

    Args:
        seq: The DNA sequence string (e.g., "AGCTNacgt").

    Returns:
        A 2D numpy array of shape (sequence_length, 4) containing the
        one-hot encoded representation of the sequence.
    """
    byte_values = np.frombuffer(seq.encode('latin1'), dtype=np.uint8)
    return self._lookup_table[byte_values]
