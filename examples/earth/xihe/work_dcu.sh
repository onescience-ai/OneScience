#!/bin/bash

echo "START TIME: $(date)"
module purge
##### Launch Conda #####
module load sghpcdas/25.6 
conda init bash
source ~/.bashrc
##### Activate Conda env #####
conda activate xihe
##### Launch DTK #####
module load sghpc-mpi-gcc/25.8
##### Show env #####
which python
which hipcc
##### Set DCU #####
#export HIP_VISIBLE_DEVICES=7

##### 1 DCU Train #####
python train_xihe2f.py

##### 8 DCU Train(change nproc_per_node to set DCU numbers) #####
# torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train_xihe2f.py

##### 1 DCU Inference #####
# python inference.py

##### Result and Visualization #####
# python result.py