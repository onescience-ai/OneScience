#!/bin/bash
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    ../../train.py \
    --name="ani1x_smallest" \
    --train_file="../../data/ani1x/ANI1x_cc_DFT_rc5_train" \
    --valid_file="../../data/ani1x/ANI1x_cc_DFT_rc5_val" \
    --statistics_file="../../data/ani1x/ANI1x_cc_DFT_rc5_statistics.json" \
    --E0s="{1: -13.62222753701504, 6: -1029.4130839658328, 7: -1484.8710358098756, 8: -2041.8396277138045}" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=64 \
    --max_L=0 \
    --correlation=3 \
    --r_max=4.5 \
    --forces_weight=1000 \
    --energy_weight=40 \
    --weight_decay=1e-7 \
    --clip_grad=1.0 \
    --batch_size=128 \
    --valid_batch_size=128 \
    --max_num_epochs=500 \
    --scheduler_patience=20 \
    --patience=50 \
    --eval_interval=1 \
    --ema \
    --swa \
    --start_swa=250 \
    --swa_lr=0.00025 \
    --swa_forces_weight=10 \
    --num_workers=32 \
    --error_table='PerAtomMAE' \
    --default_dtype="float64"\
    --device=cuda \
    --seed=123 \
    --restart_latest \
    --distributed \
    --save_cpu
