#!/bin/bash
# ============================================================
# Earth 模型公共环境配置
# 用法: 在各模型的 work_dcu.sh / work_slurm.sh 中 source ../env.sh
# ============================================================

echo "START TIME: $(date)"
module purge

##### Load Conda & DTK #####
module load sghpcdas/25.6
source /work2/share/sghpc_sdk/Linux_x86_64/25.6/das/conda/bin/activate
# conda init bash
source ~/.bashrc
module load sghpc-mpi-gcc/26.3

##### Activate env #####
conda activate py311
source ../../../env.sh

##### Verify env #####
which python
which hipcc

##### Set DCU #####
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
