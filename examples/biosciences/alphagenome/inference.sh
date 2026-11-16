#!/bin/bash

source ../../../env.sh
export PYTHONPATH=$(pwd):$PYTHONPATH
export DATA_ROOT_DIR=${ONESCIENCE_DATASETS_DIR}/AlphaGenome
export MODEL_ROOT_DIR=${ONESCIENCE_MODELS_DIR}/AlphaGenome

python run_inference.py \
    --fasta_path ${DATA_ROOT_DIR}/reference/HOMO_SAPIENS/GRCh38.p13.genome.fa \
    --model_dir ${MODEL_ROOT_DIR}/alphagenome-all-folds \
    --chromosome chr19 \
    --start 10587331 \
    --end 11635907 \
    --output_dir ./outputs
