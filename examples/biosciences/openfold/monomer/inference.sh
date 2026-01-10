#!/bin/bash

export FASTA_DIR=./monomer/fasta_dir
export OUTPUT_DIR=./monomer/
export PRECOMPUTED_ALIGNMENT_DIR=./monomer/alignments
export MMCIF_DIR=${ONESCIENCE_DATASETS_DIR}/alphafold2.3.0/pdb_mmcif/mmcif_files/ # UPDATE with path to your mmcifs directory 

python3 ./run_pretrained_openfold.py $FASTA_DIR \
  $MMCIF_DIR \
  --output_dir $OUTPUT_DIR \
  --config_preset model_1_ptm \
  --model_device "cuda:0" \
  --data_random_seed 42 \
  --use_precomputed_alignments $PRECOMPUTED_ALIGNMENT_DIR \
  --openfold_checkpoint_path ${ONESCIENCE_MODELS_DIR}/OpenFold/finetuning_ptm_2.pt