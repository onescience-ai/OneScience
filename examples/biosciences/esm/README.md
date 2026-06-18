# ESM Examples

This directory contains the ESM examples migrated into OneScience. Use `import onescience.models.esm as esm` or `from onescience.models.esm...` in new code. The upstream ESM project overview is preserved in `UPSTREAM_README.md`.

Runnable helpers live under `scripts/`, for example `python scripts/extract.py ...` from this directory. If you want to run the bundled `infer.sh` example as-is, start from the repository root because the script uses `examples/biosciences/esm/...` paths.

## Run the bundled inference example

`infer.sh` demonstrates three common ESM workflows in sequence:

* ESM-2 representation extraction with `scripts/extract.py`
* ESMFold structure prediction with `scripts/fold.py`
* ESM-1v zero-shot variant scoring with `variant-prediction/predict.py`

Run it from the repository root after loading the OneScience environment variables:

```bash
cd /path/to/onescience
source env.sh
bash examples/biosciences/esm/infer.sh
```

`env.sh` sets `ONESCIENCE_MODELS_DIR`, which is required by the example commands. In the current shared deployment, ESM weights are stored under:

```bash
$ONESCIENCE_MODELS_DIR/esm_models/
```

## ESMFold structure prediction

The ESMFold step in `infer.sh` is:

```bash
python examples/biosciences/esm/scripts/fold.py \
  -i examples/biosciences/esm/data/few_proteins.fasta \
  -o /tmp/esmfold_pdb_out \
  --model-dir $ONESCIENCE_MODELS_DIR/esm_models/
```

This command loads the `esmfold_v1` model and predicts one structure for each FASTA entry. With the bundled `data/few_proteins.fasta`, the output directory will contain one PDB file per sequence, named from the FASTA header, for example `UniRef50_UPI0003108055.pdb`.

### Important arguments

* `-i/--fasta`: input FASTA file.
* `-o/--pdb`: output directory for predicted PDB files.
* `-m/--model-dir`: parent directory used by `torch.hub` to locate pretrained ESM assets. It should contain `checkpoints/esmfold_3B_v1.pt`. In the shared OneScience setup this is `$ONESCIENCE_MODELS_DIR/esm_models/`.
* `--num-recycles`: number of structure refinement recycles. If omitted, the training default is used.
* `--max-tokens-per-batch`: maximum total sequence length per forward pass. Lower this if short-sequence batches run out of memory. Set it to `0` to effectively disable batching.
* `--chunk-size`: axial attention chunk size. Smaller values such as `128`, `64`, or `32` reduce memory usage at the cost of speed.
* `--cpu-only`: run entirely on CPU.
* `--cpu-offload`: keep part of the model on CPU RAM while using GPU for inference. This is useful for longer sequences when GPU memory is tight.

### Output and logs

For each sequence, `scripts/fold.py` writes:

* a `{header}.pdb` structure file into the output directory
* an INFO log line containing sequence length, mean `pLDDT`, `pTM`, elapsed time, and overall completion progress

If a FASTA record contains multiple chains, separate them with `:` in a single sequence entry. `scripts/fold.py` sorts sequences by length before batching, so short sequences may be inferred together for better throughput.

## What's in this directory

* The notebooks are introduced and summarized in `UPSTREAM_README.md`
* `scripts/` contains runnable ESM example helpers such as representation extraction and ESMFold inference.
* `atlas/` preserves the upstream ESM Atlas bulk-download documentation and file lists.
* `data/some_proteins.fasta` and its smaller version, `data/few_proteins.fasta` are a random selection of UniRef50 sequences used in the second example of `UPSTREAM_README.md`
* `data/1a3a_1_A.a3m`, `data/1xcr_1_A.a3m`, `data/5ahw_1_A.a3m` are MSAs distributed with trRosetta, used in `contact_prediction.ipynb`
* `data/P62593.fasta` is introduced and used in `sup_variant_prediction.ipynb`
* Example MSAs genereated in the same way as the MSAs used for MSA Transformer pre-training:
  - `data/UniRef50_E9K9Y4.a3m`, `data/UniRef50_UPI0003108055.a3m`, `data/UniRef50_UPI0003674933.a3m`, from the same sequences as trRosetta:
  `data/hhblits_uniclust_2017_10_1a3a_1_A.a3m`, `data/hhblits_uniclust_2017_10_1xcr_1_A.a3m`, `data/hhblits_uniclust_2017_10_5ahw_1_A.a3m`.
  - Generated with: `hhblits -i UniRef50_$id.fas -oa3m UniRef50_$id.a3m -n 3 -d /uniclust30_2017_10/uniclust30_2017_10`.
* `esm2_infer_fairscale_fsdp_cpu_offloading.py` shows how to load the ESM-2 15B model with Fairscale's FSDP's CPU offloading capability on a single GPU
