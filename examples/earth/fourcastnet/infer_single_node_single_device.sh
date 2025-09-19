#!/bin/bash

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

conda activate fourcastnet
module load compiler/dtk/25.04

which python
which hipcc

python inference.py

