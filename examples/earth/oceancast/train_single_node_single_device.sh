#!/bin/bash

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

source ~/conda.env
conda activate pangu_weather
module load compiler/dtk/25.04

which python
which hipcc

python train_oceancast.py
