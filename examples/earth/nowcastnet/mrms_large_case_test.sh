#!/bin/bash

python -u run.py \
    --worker 1 \
    --device cuda:0 \
    --cpu_worker 2 \
    --dataset_name radar \
    --dataset_path ./data/dataset/mrms/large_figure \
    --pretrained_model ./data/checkpoints/mrms_model.ckpt \
    --gen_frm_dir ./results/us_large/ \
    --num_save_samples 10 \
    --model_name NowcastNet \
    --img_height 1024 \
    --img_width 1024 \
    --case_type large \
    --img_ch 2 \
    --input_length 9 \
    --total_length 29 \
    --batch_size 1 