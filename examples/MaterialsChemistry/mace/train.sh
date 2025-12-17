#!/bin/bash
mkdir -p ./MACE_models
python train.py \
  --model="MACE" \
  --name="mace01" \
  --model_dir="./MACE_models" \
  --seed=123 \
  --device="cuda" \
  --r_max=4.0 \
  --batch_size=10 \
  --max_num_epochs=100 \
  --train_file="./data/solvent_xtb_train_200.xyz" \
  --test_file="./data/solvent_xtb_test.xyz" \
  --valid_fraction=0.10 \
  --energy_key="energy_xtb" \
  --forces_key="forces_xtb" \
  --E0s=average \
  --swa