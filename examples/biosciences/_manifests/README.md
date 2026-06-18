# Bio Inference Manifests

This directory contains lightweight request and run manifests for OneScience
biosciences examples.

- `model_requests/` contains per-model request examples aligned with
  `examples/biosciences/<model>/` entrypoints.
- `inference_run_manifest.yaml` is a normalized run record template for
  generated or submitted inference jobs.
- `tools/validate_bio_inference_manifest.py` validates required manifest fields.
- `tools/check_inference_outputs.py` checks whether output directories contain
  expected artifacts for a model family.
- `contract.json` declares the stable asset contract consumed by oneskills.

Set `ONESCIENCE_ROOT` to the repository root when resolving request examples
outside this checkout.
