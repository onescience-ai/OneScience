#!/bin/bash
#SBATCH -J nano_l0
#SBATCH -p hpctest03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 前提：使用 torchrun 等自带启动器
#SBATCH --cpus-per-task=128         # 增加 CPU 核心数，防止卡脖子
#SBATCH --gres=dcu:8
#SBATCH --output=%j.log              # 修复日志命名问题
#SBATCH --time=24:00:00

module purge
source ~/.bashrc
conda activate chem_py11_25043
export PYTHONNOUSERSITE=1
module load sghpc-mpi-gcc/26.3

source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 建议在运行前打印一下当前节点信息和可见显卡，方便排错
echo "========================================="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================="

# 训练前检查：打印一次当前 DCU 状态，确保显存是空的
echo ">>> Pre-training DCU Status (Checking RAM):"
hy-smi
echo "-----------------------------------------"

#bash train_multi_l_0.sh
bash train.sh

