"""AlphaFold 3 featurisation pipeline."""

import datetime
import time
from collections.abc import Sequence

import numpy as np

from onescience.flax_models.alphafold3.common import folding_input
from onescience.flax_models.alphafold3.constants import chemical_components
from onescience.flax_models.alphafold3.model import features
from onescience.flax_models.alphafold3.model.pipeline import pipeline


def validate_fold_input(fold_input: folding_input.Input):
    """Validates the fold input contains MSA and templates for featurisation."""
    for i, chain in enumerate(fold_input.protein_chains):
        if chain.unpaired_msa is None:
            raise ValueError(
                f"Protein chain {i + 1} is missing unpaired MSA.")
        if chain.paired_msa is None:
            raise ValueError(
                f"Protein chain {i + 1} is missing paired MSA.")
        if chain.templates is None:
            raise ValueError(
                f"Protein chain {i + 1} is missing Templates.")
    for i, chain in enumerate(fold_input.rna_chains):
        if chain.unpaired_msa is None:
            raise ValueError(
                f"RNA chain {i + 1} is missing unpaired MSA.")


def featurise_input(
    fold_input: folding_input.Input,
    ccd: chemical_components.Ccd,
    buckets: Sequence[int] | None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    verbose: bool = False,
) -> Sequence[features.BatchDict]:
    """Featurise the folding input.

    Args:
      fold_input: The input to featurise.
      ccd: The chemical components dictionary.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.
      ref_max_modified_date: Optional maximum date that controls whether to allow
        use of model coordinates for a chemical component from the CCD if RDKit
        conformer generation fails and the component does not have ideal
        coordinates set. Only for components that have been released before this
        date the model coordinates can be used as a fallback.
      conformer_max_iterations: Optional override for maximum number of iterations
        to run for RDKit conformer search.
      verbose: Whether to print progress messages.

    Returns:
      A featurised batch for each rng_seed in the input.
    """
    validate_fold_input(fold_input)

    # Set up data pipeline for single use.
    data_pipeline = pipeline.WholePdbPipeline(
        config=pipeline.WholePdbPipeline.Config(
            buckets=buckets,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations,
        ),
    )

    batches = []
    for rng_seed in fold_input.rng_seeds:
        featurisation_start_time = time.time()
        if verbose:
            print(f"Featurising data with seed {rng_seed}.")
        batch = data_pipeline.process_item(
            fold_input=fold_input,
            ccd=ccd,
            random_state=np.random.RandomState(rng_seed),
            random_seed=rng_seed,
        )
        if verbose:
            print(
                f"Featurising data with seed {rng_seed} took"
                f" {time.time() - featurisation_start_time:.2f} seconds."
            )
        batches.append(batch)

    return batches
