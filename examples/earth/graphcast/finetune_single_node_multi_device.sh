#!/bin/bash
unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

source ~/conda.env
conda activate graphcast
module load compiler/dtk/25.04

which python
which hipcc

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 finetune_graphcast.py
