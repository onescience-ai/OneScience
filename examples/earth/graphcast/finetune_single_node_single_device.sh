#!/bin/bash

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

source ~/conda.env
conda activate graphcast
module load compiler/dtk/25.04

which python
which hipcc

python finetune_graphcast.py
