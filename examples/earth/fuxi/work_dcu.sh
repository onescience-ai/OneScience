#!/bin/bash

echo "START TIME: $(date)"
module purge
##### Launch Conda #####
module load sghpcdas/25.6 
conda init bash
source ~/.bashrc

##### Activate Conda env #####
conda activate earth

##### Launch DTK #####
module load sghpc-mpi-gcc/25.8

##### Show env #####
which python
which hipcc

##### Set DCU #####
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

###########  Fuxi contains 7 steps, train & finetune progress, it must be done step by step  ###########
###########  7 steps are: 
###########     1. train.py (fuxi-base model)
###########     2. train_short.py (train fuxi-short model, must have base-model weight)
###########     3. inference.py short (get fuxi-short inference, which will be used as fuxi-medium input)
###########     4. train_medium.py (train fuxi-medium model, must have short-model weight)
###########     5. inference.py medium (get fuxi-medium inference, which will be used as fuxi-long input)
###########     6. train_long.py (train fuxi-long model, must have medium-model weight)
###########     7. inference.py long (get fuxi-long inference)
###########     Notice that, 'python inference.py base' can be run anytime after get base-model weights from process 1.

##### train & finetune #####
# 1-DCU
python train_base.py 
# python train_short.py
# python train_medium.py
# python train_long.py

# 8-DCU
# torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train_base.py
# torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train_short.py
# torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train_medium.py
# torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train_long.py

##### inference (Notice that each inference must be done after train) #####
# python inference.py base
# python inference.py short
# python inference.py medium
# python inference.py long

##### Result and Visualization #####
# python result.py base
# python result.py short
# python result.py medium
# python result.py long