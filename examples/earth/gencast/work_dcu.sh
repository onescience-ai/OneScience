#!/bin/bash

echo "START TIME: $(date)"
module purge
# ##### Launch Conda #####
module load sghpcdas/25.6 
conda init bash
source ~/.bashrc

# ##### Activate Conda env #####
conda activate gencast

# ##### Launch DTK #####
module load sghpc-mpi-gcc/25.8

##### Launch env #####
source ../../../env.sh

##### Show env #####
which python
which hipcc
##### Set DCU #####
export HIP_VISIBLE_DEVICES=0

##### 1 DCU Train #####
python train.py

##### 1 DCU Inference #####
# python inference.py

##### Result and Visualization #####
# python result.py