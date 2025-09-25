#!/bin/bash
#SBATCH -J evo2_for_onescience
#SBATCH -p k100ai
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH -o evo2/logs%j.out       

source ~/dtk/dtk-25.04.1/env.sh
source ~/dtk/dtk-25.04.1/cuda/env.sh
module load compilers/gcc/12.2.0
source ~/conda.env
conda activate test-evo2env
unset ROCBLAS_TENSILE_LIBPATH 

DEVICES=${SLURM_GPUS_PER_NODE:-4}
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" 

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# 第一个节点的地址
master_addr=${nodes_array[0]}

# 主节点的端口（可以根据需要调整）
master_port=29500

# 在每个节点上启动 torchrun
echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$master_addr
echo master_port=$master_port

srun train_evo2.sh
