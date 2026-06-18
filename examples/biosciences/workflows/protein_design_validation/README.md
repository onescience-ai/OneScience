# Protein Design Validation Workflow

This workflow request describes a staged OneScience example that connects:

- RFdiffusion backbone generation
- ProteinMPNN sequence design
- SimpleFold structure validation

`request.yaml` is an implementation contract for generating runnable pipeline
scripts. `tools/setup_esm_torchhub.sh` prepares offline ESM and SimpleFold
assets used by the validation stage.
