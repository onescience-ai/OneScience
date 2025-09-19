#!/bin/bash
#SBATCH -J evo2_for_onescience
#SBATCH -p newlarge
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:8
#SBATCH -o evo2/logs%j.out       

DIRS=(
    "./lightning_logs"
    "./results"
)

for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "Del Files: $DIR"
        rm -rf "$DIR"
    else
        echo "Files Not Exist: $DIR"
    fi
done

module load sghpc-mpi-gcc/25.6
source /public/home/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate evo2-biao
unset ROCBLAS_TENSILE_LIBPATH 

DEVICES=${SLURM_GPUS_PER_NODE:-8}
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" 

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
