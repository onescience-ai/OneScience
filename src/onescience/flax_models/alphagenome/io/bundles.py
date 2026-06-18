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

"""Bundle names and their associated keys and resolutions."""

import enum
import tensorflow as tf


class BundleName(enum.Enum):
  """Bundle names."""

  ATAC = 'atac'
  DNASE = 'dnase'
  PROCAP = 'procap'
  CAGE = 'cage'
  RNA_SEQ = 'rna_seq'
  CHIP_TF = 'chip_tf'
  CHIP_HISTONE = 'chip_histone'
  CONTACT_MAPS = 'contact_maps'
  SPLICE_SITES_CLASSIFICATION = 'splice_sites'
  SPLICE_SITES_USAGE = 'splice_site_usage'
  SPLICE_SITES_JUNCTION = 'splice_junctions'
  SPLICE_SITES_POSITIONS = 'splice_site_positions'

  def get_dtypes(self) -> dict[str, tf.DType]:
    """Returns the keys and dtypes for the given bundle."""
    match self:
      case BundleName.ATAC:
        return {'atac': tf.bfloat16, 'atac_mask': tf.bool}
      case BundleName.DNASE:
        return {'dnase': tf.bfloat16, 'dnase_mask': tf.bool}
      case BundleName.PROCAP:
        return {'procap': tf.bfloat16, 'procap_mask': tf.bool}
      case BundleName.CAGE:
        return {'cage': tf.bfloat16, 'cage_mask': tf.bool}
      case BundleName.RNA_SEQ:
        return {
            'rna_seq': tf.bfloat16,
            'rna_seq_mask': tf.bool,
            'rna_seq_strand': tf.int32,
        }
      case BundleName.CHIP_TF:
        return {'chip_tf': tf.float32, 'chip_tf_mask': tf.bool}
      case BundleName.CHIP_HISTONE:
        return {'chip_histone': tf.float32, 'chip_histone_mask': tf.bool}
      case BundleName.CONTACT_MAPS:
        return {'contact_maps': tf.float32}
      case BundleName.SPLICE_SITES_CLASSIFICATION:
        return {'splice_sites': tf.bool}
      case BundleName.SPLICE_SITES_USAGE:
        return {'splice_site_usage': tf.float16}
      case BundleName.SPLICE_SITES_JUNCTION:
        return {'splice_junctions': tf.float32}
      case BundleName.SPLICE_SITES_POSITIONS:
        return {'splice_site_positions': tf.int32}
      case _:
        raise ValueError(f'Unknown bundle name: {self}')

  def get_resolution(self) -> int:
    """Returns the resolutions for the given bundle."""
    match self:
      case (
          BundleName.ATAC
          | BundleName.DNASE
          | BundleName.PROCAP
          | BundleName.CAGE
          | BundleName.RNA_SEQ
          | BundleName.SPLICE_SITES_CLASSIFICATION
          | BundleName.SPLICE_SITES_USAGE
          | BundleName.SPLICE_SITES_JUNCTION
          | BundleName.SPLICE_SITES_POSITIONS
      ):
        return 1
      case BundleName.CHIP_TF | BundleName.CHIP_HISTONE:
        return 128
      case BundleName.CONTACT_MAPS:
        return 2_048
      case _:
        raise ValueError(f'Unknown bundle name: {self}')
