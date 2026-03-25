#!/bin/bash
#mkdir -p ./MACE_models
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source ../../../../../env.sh
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    ../../train.py \
    --name="nanotube_l_2" \
    --train_file="${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/nanotube/nanotube_large.xyz" \
    --valid_fraction=0.05 \
    --test_file="${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/nanotube/nanotube_test.xyz" \
    --E0s="average" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=256 \
    --max_L=2 \
    --correlation=3 \
    --r_max=5.0 \
    --forces_weight=1000 \
    --energy_weight=10 \
    --energy_key="Energy" \
    --forces_key="forces" \
    --batch_size=4 \
    --valid_batch_size=8 \
    --max_num_epochs=100 \
    --start_swa=60 \
    --scheduler_patience=5 \
    --patience=15 \
    --eval_interval=10 \
    --ema \
    --swa \
    --swa_forces_weight=10 \
    --error_table='PerAtomMAE' \
    --default_dtype="float64"\
    --device=cuda \
    --distributed \
    --seed=123 \
    --restart_latest \
    --save_cp
