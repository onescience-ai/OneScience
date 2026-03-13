#!/bin/bash
python ../../train.py \
  --model="MACE" \
  --name="mace01" \
  --seed=123 \
  --device="cuda" \
  --r_max=4.0 \
  --batch_size=10 \
  --max_num_epochs=10 \
  --train_file="/public/onestore/onedatasets/MaterialsChemistry/examples/DMC/solvent_xtb_train_200.xyz" \
  --test_file="/public/onestore/onedatasets/MaterialsChemistry/examples/DMC/solvent_xtb_test.xyz" \
  --valid_fraction=0.10 \
  --energy_key="energy_xtb" \
  --forces_key="forces_xtb" \
  --E0s=average \
  --swa
