#!/bin/bash

set -e

PATH_FOR_TRAINING_DATA="/public/onestore/onedatasets/proteinmpnn/pdb_2021aug02_sample/"

python ./training.py \
           --path_for_outputs "./exp_020" \
           --path_for_training_data $PATH_FOR_TRAINING_DATA \
           --num_examples_per_epoch 1000 \
           --save_model_every_n_epochs 50
