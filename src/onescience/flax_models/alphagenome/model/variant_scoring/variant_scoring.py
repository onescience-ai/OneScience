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

"""Base class for variant scorers."""

import abc
from collections.abc import Mapping
import functools
from typing import Generic, TypeVar

from onescience.flax_models.alphagenome._sdk import typing
from onescience.flax_models.alphagenome._sdk.data import genome
from onescience.flax_models.alphagenome._sdk.models import dna_output
import anndata
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32, Int32, PyTree  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd


VariantMaskT = TypeVar('VariantMaskT')
VariantMetadataT = TypeVar('VariantMetadataT')
VariantSettingsT = TypeVar('VariantSettingsT')

ScoreVariantOutput = Mapping[str, jax.Array | np.ndarray]
ScoreVariantResult = Mapping[str, np.ndarray]

ScoreVariantInput = Mapping[
    dna_output.OutputType, PyTree[Float32[Array, '...'] | Int32[Array, '...']]
]


@typing.jaxtyped
def create_anndata(
    scores: Float32[np.ndarray, 'G T'],
    *,
    obs: pd.DataFrame | None,
    var: pd.DataFrame,
) -> anndata.AnnData:
  """Helper function for creating AnnData objects."""
  var = var.copy()
  # We explicitly cast the dataframe indices to str to avoid
  # ImplicitModificationWarning being logged over and over again.
  var.index = var.index.map(str)

  if obs is not None:
    obs = obs.copy()
    obs.index = obs.index.map(str).astype(str)
  return anndata.AnnData(np.ascontiguousarray(scores), obs=obs, var=var)


def get_resolution(output_type: dna_output.OutputType):
  match output_type:
    case dna_output.OutputType.ATAC:
      return 1
    case dna_output.OutputType.CAGE:
      return 1
    case dna_output.OutputType.DNASE:
      return 1
    case dna_output.OutputType.RNA_SEQ:
      return 1
    case dna_output.OutputType.CHIP_HISTONE:
      return 128
    case dna_output.OutputType.CHIP_TF:
      return 128
    case dna_output.OutputType.SPLICE_SITES:
      return 1
    case dna_output.OutputType.SPLICE_SITE_USAGE:
      return 1
    case dna_output.OutputType.SPLICE_JUNCTIONS:
      return 1
    case dna_output.OutputType.CONTACT_MAPS:
      return 2048
    case dna_output.OutputType.PROCAP:
      return 1
    case _:
      raise ValueError(f'Unknown output type: {output_type}.')


class VariantScorer(
    Generic[VariantMaskT, VariantMetadataT, VariantSettingsT],
    metaclass=abc.ABCMeta,
):
  """Abstract class for variant scorers."""

  @abc.abstractmethod
  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      settings: VariantSettingsT,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[VariantMaskT, VariantMetadataT]:
    """Returns masks and metadata for the given interval, variant and metadata.

    The generated masks and metadata will be passed to `score_variant` and
    `finalize_variant` respectively.

    Args:
      interval: The interval to score.
      variant: The variant to extract the masks/metadata for.
      settings: The variant scorer settings.
      track_metadata: The track metadata required to finalize the variant. These
        will be passed into the `finalize_variants` function.

    Returns:
      A tuple of (masks, metadata), where:
        masks: The masks required to score the variant, such as gene or TSS or
          strand masks. These will be passed into the jitted `score_variants`
          function.
        metadata: The metadata required to finalize the variant. These will
          be passed into the `finalize_variants` function.

      The formats/shapes of masks and metadata will vary across variant scorers
      depending on their individual needs.
    """

  @abc.abstractmethod
  def score_variant(
      self,
      ref: ScoreVariantInput,
      alt: ScoreVariantInput,
      *,
      masks: VariantMaskT,
      settings: VariantSettingsT,
      variant: genome.Variant | None = None,
      interval: genome.Interval | None = None,
  ) -> ScoreVariantOutput:
    """Generates a score per track for the provided ref/alt predictions.

    Args:
      ref: Reference predictions.
      alt: Alternative predictions.
      masks: The masks for scoring the variant.
      settings: The variant scorer settings.
      variant: The variant to score.
      interval: The interval to score.

    Returns:
      Dictionary of scores to be passed to `finalize_variant`.
    """

  @abc.abstractmethod
  def finalize_variant(
      self,
      scores: ScoreVariantResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: VariantMetadataT,
      settings: VariantSettingsT,
  ) -> anndata.AnnData:
    """Returns finalized scores for the given scores and metadata.

    Args:
      scores: Dictionary of scores generated from `score_variant` function.
      track_metadata: Metadata describing the tracks for each output_type.
      mask_metadata: Metadata describing the masks.
      settings: The variant scorer settings.

    Returns:
      A VariantOutputType object containing the final variant outputs. The
      entries will vary across scorers depending on their individual needs.
    """


@typing.jaxtyped
def align_alternate(
    alt: Float32[Array | np.ndarray, 'S T'],
    variant: genome.Variant,
    interval: genome.Interval,
) -> Float32[Array, 'S T']:
  """Aligns ALT predictions to match the REF allele's sequence length.

  This function adjusts the `alt` prediction array to account for indels
  (insertions or deletions) present in the `variant`.

  For insertions, the function summarizes the inserted region by taking the
  maximum value across the alternate bases and pads the end with zeros to
  maintain the original sequence length.

  For deletions, zero signal is inserted at the locations corresponding to the
  deleted bases in the reference.

  Args:
    alt: The ALT allele predictions, shape [sequence_length, num_tracks].
    variant: The variant containing the indel information.
    interval: The genomic interval.

  Returns:
    The aligned ALT predictions, shape [sequence_length, num_tracks].
  """

  insertion_length = len(variant.alternate_bases) - len(variant.reference_bases)
  deletion_length = -insertion_length
  variant_start_in_vector = variant.start - interval.start
  # We assume that variants are left-aligned, and that insertions/deletions
  # for multi-change variants occur at the end of the variant.
  # We only need to align that insertion/deletion portion.
  variant_start_in_vector += (
      min(len(variant.reference_bases), len(variant.alternate_bases)) - 1
  )
  original_length = alt.shape[0]

  # Summarize potential insertions by computing the maximum score across
  # alternate bases.

  @functools.partial(jax.jit, static_argnames=['insertion_length'])
  def _apply(alt, insertion_length: int):
    if insertion_length > 0:
      pool_alt_past_ref = jnp.max(
          alt[
              variant_start_in_vector : variant_start_in_vector
              + insertion_length
              + 1
          ],
          axis=0,
          keepdims=True,
      )
      alt = jnp.concatenate(
          [
              alt[:variant_start_in_vector],
              pool_alt_past_ref,
              alt[(variant_start_in_vector + insertion_length + 1) :],
              jnp.zeros((insertion_length, alt.shape[1])),
          ],
          axis=0,
      )
      # Truncate to the original sequence length in case the alt insertion
      # spills over the original sequence length. This happens only for
      # insertions longer than half the interval.
      alt = alt[:original_length]
    elif deletion_length > 0:
      # Handle potential deletions by inserting zero signal at deletion
      # locations.
      alt = jnp.concatenate(
          [
              alt[: (variant_start_in_vector + 1)],
              jnp.zeros((deletion_length, alt.shape[1])),
              alt[(variant_start_in_vector + 1) :],
          ],
          axis=0,
      )
      alt = alt[:original_length]
    return alt

  return _apply(alt, insertion_length)
