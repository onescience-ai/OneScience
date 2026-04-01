#!/bin/bash
#SBATCH -J nano_l2
#SBATCH -p hpctest03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 前提：使用 torchrun 等自带启动器
#SBATCH --cpus-per-task=128         # 增加 CPU 核心数，防止卡脖子
#SBATCH --gres=dcu:8
#SBATCH --output=%j.log              # 修复日志命名问题
#SBATCH --time=24:00:00
##SBATCH --exclude=a09r1n17
##SBATCH --nodelist=a10r3n10          # 锁定你扫描到的干净节点

module purge
source ~/.bashrc
conda activate chem_py11_25043 
module load sghpc-mpi-gcc/26.3

source /public/software/compiler/dtk-25.04.2/cuda/env.sh
export LD_LIBRARY_PATH="/public/home/easyscience2024/.conda/envs/chem_py11_25043/lib/python3.11/site-packages/fastpt/torch/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


# 建议在运行前打印一下当前节点信息和可见显卡，方便排错
echo "========================================="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================="

# 训练前检查：打印一次当前 DCU 状态，确保显存是空的
echo ">>> Pre-training DCU Status (Checking RAM):"
hy-smi
echo "-----------------------------------------"

bash train_nanotube_l_2.sh
#bash train_nanotube_l_0.sh
