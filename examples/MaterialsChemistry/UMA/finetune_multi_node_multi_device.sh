#!/bin/bash
#SBATCH -p k100ai
#SBATCH -N 2
#SBATCH --gres=dcu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -J uma
#SBATCH -o ./log/%j.out # 标准输出日志文件保存路径
#SBATCH -e ./log/%j.err # 标准错误日志文件保存路径


module purge
module load mpi/hpcx/2.12.0/gcc-8.3.1
module load compiler/dtk/25.04
source /work/home/onescience2025/anaconda3/bin/activate
conda activate fairchemtest

unset ROCBLAS_TENSILE_LIBPATH

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export NCCL_PROTO=Simple
which python
which hipcc
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
master_addr=${nodes_array[0]}
master_port=29504
echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$master_addr
echo master_port=$master_port



# === 训练参数 ===
CONFIG=./dataset/oc20/uma_oc20_finetune/uma_sm_finetune_template.yaml 

srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES torchrun \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$master_addr:$master_port \
  env_lauch.py \
  -c ${CONFIG}





