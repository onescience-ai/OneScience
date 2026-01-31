#!/bin/bash
mkdir -p ./MACE_models
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
   train.py \
  --name='mace03' \
  --foundation_model=./MACE_models/mace-mpa-0-medium.model\
  --model='MACE' \
  --model_dir="./MACE_models" \
  --num_channels=32 \
  --max_L=0 \
  --r_max=4.0 \
  --train_file='./data/solvent_xtb_train_200.xyz' \
  --valid_fraction=0.10 \
  --test_file='./data/solvent_xtb_test.xyz' \
  --energy_key='energy_xtb' \
  --forces_key='forces_xtb' \
  --batch_size=10 \
  --max_num_epochs=100 \
  --swa \
  --seed=123 \
  --distributed \
  --device='cuda' \
  --num_workers=8