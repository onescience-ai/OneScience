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

"""Runs AlphaGenome track prediction evaluation.

By default this script refuses to download model weights. Pass --model_dir to
load a local Orbax checkpoint, or explicitly pass --allow_download=true to use
the original Kaggle Hub path.
"""

import functools
import pathlib
import pprint

from absl import app
from absl import flags
from absl import logging
from onescience.flax_models.alphagenome._sdk.data import fold_intervals
from onescience.flax_models.alphagenome._sdk.models import dna_output
from onescience.flax_models.alphagenome._sdk.models import dna_model as dna_model_types
from onescience.flax_models.alphagenome.evals.track_prediction import (
    load_model as load_model_from_kaggle,
)
from onescience.flax_models.alphagenome.evals import regression_metrics
from onescience.flax_models.alphagenome.io.bundles import BundleName
from onescience.flax_models.alphagenome.io.dataset import get_numpy_dataset_iterator
from onescience.flax_models.alphagenome.model import dna_model as research_dna_model
from onescience.flax_models.alphagenome.model.metadata import metadata as metadata_lib
import jax
import jax.numpy as jnp
from jax import sharding
from jax.experimental import mesh_utils
import orbax.checkpoint as ocp
import pandas as pd
import tensorflow as tf


FLAGS = flags.FLAGS
PS = sharding.PartitionSpec

flags.DEFINE_string(
    "model_version",
    "FOLD_0",
    "Evaluation fold/model version. Supported values are "
    + ",".join(version.name for version in dna_model_types.ModelVersion)
    + ".",
)
flags.DEFINE_enum(
    "organism",
    "HOMO_SAPIENS",
    ["HOMO_SAPIENS", "MUS_MUSCULUS"],
    "Organism to evaluate.",
)
flags.DEFINE_string(
    "output_path",
    "./track_prediction_results.csv",
    "Output CSV path.",
)
flags.DEFINE_string(
    "model_dir",
    None,
    "Local AlphaGenome Orbax checkpoint directory. When set, model weights are "
    "loaded from this directory and no online model download is attempted.",
)
flags.DEFINE_bool(
    "allow_download",
    False,
    "Allow Kaggle Hub model download only when --model_dir is not provided.",
)
flags.DEFINE_string(
    "data_dir",
    None,
    "Optional AlphaGenome TFRecord root. If unset, the dataset loader uses its "
    "built-in default path.",
)
flags.DEFINE_list(
    "bundles",
    None,
    "Comma-separated bundle names. Defaults to all supported evaluation "
    "bundles: ATAC,CAGE,CHIP_HISTONE,CHIP_TF,DNASE,PROCAP,RNA_SEQ.",
)

_DEFAULT_EVAL_BUNDLES = [
    BundleName.ATAC,
    BundleName.CAGE,
    BundleName.CHIP_HISTONE,
    BundleName.CHIP_TF,
    BundleName.DNASE,
    BundleName.PROCAP,
    BundleName.RNA_SEQ,
]


def _parse_model_version(value: str) -> dna_model_types.ModelVersion:
    normalized = value.replace("-", "_").upper()
    if normalized.startswith("ALPHAGENOME_"):
        normalized = normalized.removeprefix("ALPHAGENOME_")
    try:
        return dna_model_types.ModelVersion[normalized]
    except KeyError as exc:
        valid = ", ".join(version.name for version in dna_model_types.ModelVersion)
        raise ValueError(
            f"Unsupported --model_version={value!r}. Valid values: {valid}."
        ) from exc


def _resolve_local_model_dir(path: str) -> pathlib.Path:
    model_dir = pathlib.Path(path).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Local AlphaGenome model directory does not exist: {model_dir}"
        )
    if not (model_dir / "_CHECKPOINT_METADATA").exists():
        raise FileNotFoundError(
            "The model directory does not look like an Orbax checkpoint. "
            f"Missing: {model_dir / '_CHECKPOINT_METADATA'}"
        )
    return model_dir


def load_model_from_local_checkpoint(
    checkpoint_path: str,
    organism: dna_model_types.Organism,
):
    """Loads eval params/state/predict_fn like run_inference.py does."""
    checkpoint_path = _resolve_local_model_dir(checkpoint_path)
    logging.info("Loading AlphaGenome model from local checkpoint: %s", checkpoint_path)
    metadata = {organism: metadata_lib.load(organism)}
    init_fn, apply_fn, _ = research_dna_model.create_model(metadata)
    dna_sequence_shape = jax.ShapeDtypeStruct((1, 2048, 4), dtype=jnp.float32)
    organism_index_shape = jax.ShapeDtypeStruct((1,), dtype=jnp.int32)
    target_shapes = jax.eval_shape(
        init_fn,
        jax.random.PRNGKey(0),
        dna_sequence_shape,
        organism_index_shape,
    )
    params, state = ocp.StandardCheckpointer().restore(
        str(checkpoint_path),
        target=target_shapes,
        strict=True,
    )

    @jax.jit
    def predict(params, state, dna_sequence, organism_index):
        predictions = apply_fn(params, state, dna_sequence, organism_index)
        return research_dna_model.extract_predictions(predictions)

    return params, state, predict


def _load_model(
    model_version: dna_model_types.ModelVersion,
    organism: dna_model_types.Organism,
):
    if FLAGS.model_dir:
        return load_model_from_local_checkpoint(FLAGS.model_dir, organism)
    if FLAGS.allow_download:
        logging.warning(
            "No --model_dir was provided; downloading model weights from "
            "Kaggle Hub because --allow_download=true."
        )
        return load_model_from_kaggle(model_version)
    raise ValueError(
        "No --model_dir was provided. Refusing to download model weights. "
        "Pass --model_dir /path/to/alphagenome-all-folds, or explicitly pass "
        "--allow_download=true to use Kaggle Hub."
    )


def _resolve_data_dir(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("gs://"):
        return path
    data_dir = pathlib.Path(path).expanduser()
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Local AlphaGenome dataset directory does not exist: {data_dir}"
        )
    return str(data_dir)


def _mesh_context(mesh):
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return mesh


def create_eval_step(predict_fn, bundles):
    """Returns a JAX-version-compatible eval step."""

    @jax.jit
    def eval_step(params, state, batch):
        predictions = predict_fn(
            params,
            state,
            batch.dna_sequence,
            batch.organism_index,
        )
        metrics_step = {}
        for bundle in bundles:
            targets_true, mask = batch.get_genome_tracks(bundle)
            targets_pred = predictions[dna_output.OutputType[bundle.name]]
            targets_pred = regression_metrics.crop_sequence_length(
                targets_pred,
                target_length=targets_true.shape[-2],
            )
            metrics_step[bundle.name] = regression_metrics.update_regression_metrics(
                targets_true,
                targets_pred,
                mask,
            )
        return metrics_step

    return eval_step


def evaluate(params, state, predict_fn, bundles, dataset_iterator):
    """Evaluates the model without relying on the library's jit decorator style."""
    devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
    mesh = jax.sharding.Mesh(devices, axis_names=("data",))
    sharding_rep = sharding.NamedSharding(mesh, PS())
    sharding_data = sharding.NamedSharding(mesh, PS("data"))

    params = jax.device_put(params, sharding_rep)
    state = jax.device_put(state, sharding_rep)

    eval_step = create_eval_step(predict_fn, bundles)
    metrics = {
        bundle.name: regression_metrics.initialize_regression_metrics()
        for bundle in bundles
    }
    num_elements = 0

    for i, (batch, _) in enumerate(dataset_iterator):
        num_elements += batch.dna_sequence.shape[0]
        if i % 5 == 1:
            finalized = pprint.pformat(
                regression_metrics.finalize_regression_metrics(metrics)
            )
            logging.info("step %d: %s", i, finalized)

        with _mesh_context(mesh):
            batch = jax.device_put(batch, sharding_data)
            step_metrics = eval_step(params, state, batch)

        step_metrics = jax.device_get(step_metrics)
        metrics = regression_metrics.reduce_regression_metrics(
            metrics,
            step_metrics,
        )

    logging.info("num_elements: %d", num_elements)
    return regression_metrics.finalize_regression_metrics(metrics)


def main(_):
    # TensorFlow is only used for data loading; keep GPUs available to JAX.
    tf.config.set_visible_devices([], "GPU")

    model_version = _parse_model_version(FLAGS.model_version)
    organism = dna_model_types.Organism[FLAGS.organism]
    data_dir = _resolve_data_dir(FLAGS.data_dir)

    logging.info("JAX devices: %s", jax.devices())
    logging.info(
        "Evaluation config: model_version=%s, organism=%s, model_dir=%s, "
        "data_dir=%s",
        model_version.name,
        FLAGS.organism,
        FLAGS.model_dir or "<download disabled>",
        data_dir or "<loader default>",
    )

    if FLAGS.bundles:
        eval_bundles = [BundleName[bundle.strip()] for bundle in FLAGS.bundles]
    else:
        eval_bundles = _DEFAULT_EVAL_BUNDLES
    logging.info("Evaluation bundles: %s", [bundle.name for bundle in eval_bundles])

    params, state, predict_fn = _load_model(model_version, organism)

    dataset_iterator = get_numpy_dataset_iterator(
        batch_size=jax.local_device_count(),
        organism=organism,
        model_version=model_version,
        subset=fold_intervals.Subset.VALID,
        bundles=eval_bundles,
        path=data_dir,
    )

    logging.info("Starting evaluation.")
    results = evaluate(
        params=params,
        state=state,
        predict_fn=predict_fn,
        bundles=eval_bundles,
        dataset_iterator=dataset_iterator,
    )

    rows = []
    for bundle_name, metrics in results.items():
        for metric_name, value in metrics.items():
            logging.info("%s / %s: %.4f", bundle_name, metric_name, value)
            rows.append({
                "bundle": bundle_name,
                "metric": metric_name,
                "value": float(value),
                "model_version": model_version.name,
                "organism": FLAGS.organism,
            })

    output_path = pathlib.Path(FLAGS.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info("Saved results to: %s", output_path)


if __name__ == "__main__":
    app.run(main)
