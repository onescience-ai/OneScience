#!/bin/bash

python -u run.py \
     --do_train True \
     --gen_pre True \
     --worker 1 \
     --cpu_worker 2 \
     --dataset_name radar \
     --dataset_path ./data/dataset/mrms/figure \
     --pretrained_model ./data/checkpoints/mrms_model.ckpt \
     --gen_frm_dir ./results/us/ \
     --num_save_samples 10 \
     --model_name NowcastNet \
     --img_height 512 \
     --img_width 512 \
     --case_type normal \
     --img_ch 2 \
     --input_length 9 \
     --total_length 29 \
     --batch_size 32 \
