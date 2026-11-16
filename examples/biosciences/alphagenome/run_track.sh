#!/bin/bash

source ../../../env.sh
export PYTHONPATH=$(pwd):$PYTHONPATH
export DATA_ROOT_DIR=${ONESCIENCE_DATASETS_DIR}/AlphaGenome
export MODEL_ROOT_DIR=${ONESCIENCE_MODELS_DIR}/AlphaGenome

python run_track_prediction_eval.py \
    --model_dir ${MODEL_ROOT_DIR}/alphagenome-all-folds \
    --model_version ALL_FOLDS \
    --data_dir ${DATA_ROOT_DIR}/v1/train \
    --output_path ./outputs_track/eval_results.csv

