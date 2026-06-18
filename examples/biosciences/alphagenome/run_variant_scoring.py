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

"""Example script for AlphaGenome variant scoring.

This version avoids failing when the model was created without annotation-backed
variant scorers such as GENE_MASK_LFC. It keeps the recommended scorers that
are actually available in the current model instance and skips the rest.
"""

from collections.abc import Sequence
import pathlib

from absl import app
from absl import flags
from absl import logging
from onescience.flax_models.alphagenome._sdk.data import genome
from onescience.flax_models.alphagenome._sdk.models import dna_model as dna_model_types
from onescience.flax_models.alphagenome._sdk.models import variant_scorers as variant_scorers_lib
import pandas as pd
from onescience.flax_models.alphagenome.model.dna_model import (
    OrganismSettings,
    create,
    create_from_kaggle,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "vcf_path",
    None,
    "Path to a VCF file. If unset, built-in demo variants are used.",
)
flags.DEFINE_string(
    "fasta_path",
    None,
    "Path to the reference genome FASTA file. Required when --model_dir is set.",
)
flags.DEFINE_string(
    "model_dir",
    None,
    "Local AlphaGenome checkpoint directory. If unset, Kaggle Hub is used.",
)
flags.DEFINE_string("output_dir", "./outputs", "Directory for CSV outputs.")
flags.DEFINE_enum(
    "organism",
    "HOMO_SAPIENS",
    ["HOMO_SAPIENS", "MUS_MUSCULUS"],
    "Target organism.",
)
flags.DEFINE_enum(
    "model_version",
    "all_folds",
    ["FOLD_0", "FOLD_1", "FOLD_2", "FOLD_3", "FOLD_4", "all_folds"],
    "Model version to download from Kaggle.",
)


DEMO_VARIANTS = [
    ("chr22:36201698:A>C", "eQTL with SuSiE PIP > 0.9 in GTEx Colon"),
    ("chr3:120280774:G>T", "caQTL in GM12878 (DNase)"),
    ("chr21:46126238:G>C", "Splice junction variant in COL6A2"),
]


def load_demo_variants() -> list[tuple[genome.Variant, str]]:
  """Builds the built-in demo variants."""
  return [
      (genome.Variant.from_str(variant_str), description)
      for variant_str, description in DEMO_VARIANTS
  ]


def load_variants_from_vcf(vcf_path: str) -> list[tuple[genome.Variant, str]]:
  """Loads variants from a VCF file."""
  variants_df = pd.read_csv(
      vcf_path,
      sep="\t",
      comment="#",
      names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
  )
  variants_with_desc = []
  for _, row in variants_df.iterrows():
    variant = genome.Variant(
        chromosome=row["CHROM"],
        position=int(row["POS"]),
        reference_bases=row["REF"],
        alternate_bases=str(row["ALT"]).split(",")[0],
    )
    description = row["ID"] if pd.notna(row["ID"]) else "unknown"
    variants_with_desc.append((variant, str(description)))
  return variants_with_desc


def resolve_variant_scorers(
    alphagenome_model,
    organism: dna_model_types.Organism,
) -> tuple[
    Sequence[variant_scorers_lib.VariantScorerTypes],
    list[str],
]:
  """Returns the recommended scorers supported by the current model."""
  recommended_scorers = list(
      variant_scorers_lib.get_recommended_scorers(organism.to_proto())
  )
  available_scorer_map = getattr(alphagenome_model, "_variant_scorers", {}).get(
      organism, {}
  )

  if not available_scorer_map:
    logging.warning(
        "Unable to inspect model variant scorers. Falling back to the full "
        "recommended scorer list."
    )
    return recommended_scorers, []

  available_base_scorers = set(available_scorer_map)
  selected_scorers = [
      scorer
      for scorer in recommended_scorers
      if scorer.base_variant_scorer in available_base_scorers
  ]
  skipped_scorers = [
      scorer.base_variant_scorer.name
      for scorer in recommended_scorers
      if scorer.base_variant_scorer not in available_base_scorers
  ]

  if not selected_scorers:
    available_names = sorted(
        base_scorer.name for base_scorer in available_base_scorers
    )
    raise ValueError(
        "No compatible recommended variant scorers are available for "
        f"{organism.name}. Available scorers: {available_names}."
    )

  return selected_scorers, skipped_scorers


def load_alphagenome_model(organism: dna_model_types.Organism):
  """Loads AlphaGenome from a local checkpoint when provided."""
  organism_settings = None
  if FLAGS.fasta_path:
    organism_settings = {
        organism: OrganismSettings(
            fasta_path=FLAGS.fasta_path,
        ),
    }

  if FLAGS.model_dir:
    if not FLAGS.fasta_path:
      raise ValueError("--fasta_path is required when using --model_dir.")
    return create(
        checkpoint_path=FLAGS.model_dir,
        organism_settings=organism_settings,
    )

  return create_from_kaggle(
      FLAGS.model_version,
      organism_settings=organism_settings,
  )


def main(_):
  output_dir = pathlib.Path(FLAGS.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  organism = dna_model_types.Organism[FLAGS.organism]

  logging.info("Loading AlphaGenome model...")
  alphagenome_model = load_alphagenome_model(organism)

  variant_scorers, skipped_scorers = resolve_variant_scorers(
      alphagenome_model, organism
  )
  logging.info(
      "Using variant scorers: %s",
      ", ".join(scorer.base_variant_scorer.name for scorer in variant_scorers),
  )
  if skipped_scorers:
    logging.warning(
        "Skipping unavailable recommended scorers: %s. This usually means the "
        "model was loaded without the required annotation resources.",
        ", ".join(skipped_scorers),
    )

  if FLAGS.vcf_path:
    logging.info("Loading variants from VCF: %s", FLAGS.vcf_path)
    variants_with_desc = load_variants_from_vcf(FLAGS.vcf_path)
  else:
    logging.info("Using built-in demo variants.")
    variants_with_desc = load_demo_variants()

  logging.info("Scoring %d variants...", len(variants_with_desc))
  all_results = []

  for variant, description in variants_with_desc:
    logging.info("Processing variant: %s (%s)", variant, description)

    interval = variant.reference_interval.resize(2**20)
    scores = alphagenome_model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=variant_scorers,
        organism=organism,
    )

    all_results.append(
        {
            "variant": str(variant),
            "description": description,
            "num_score_tables": len(scores),
            "used_variant_scorers": ",".join(
                scorer.base_variant_scorer.name for scorer in variant_scorers
            ),
            "skipped_variant_scorers": ",".join(skipped_scorers),
        }
    )

    for i, adata in enumerate(scores):
      scorer_name = str(adata.uns.get("variant_scorer", f"scorer_{i}"))
      scorer_label = scorer_name.replace(" ", "_").replace("/", "_")
      save_path = output_dir / (
          f"variant_{variant.chromosome}_{variant.position}_{scorer_label}.csv"
      )
      adata.to_df().to_csv(save_path)
      logging.info(
          "Saved score table %d: %s (shape=%s)", i, save_path.name, adata.shape
      )

  summary_path = output_dir / "variant_scoring_summary.csv"
  pd.DataFrame(all_results).to_csv(summary_path, index=False)
  logging.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
  app.run(main)
