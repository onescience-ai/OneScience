#!/bin/bash
#SBATCH -p hpctest03
#SBATCH -N 1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -J uma
#SBATCH -o %j.out # 标准输出日志文件保存路径
#SBATCH -e %j.err # 标准错误日志文件保存路径

module purge
source ~/.bashrc
conda activate chem_py11_25043
module load sghpc-mpi-gcc/26.3

source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


unset ROCBLAS_TENSILE_LIBPATH
#export NCCL_SOCKET_IFNAME=ib0
#export NCCL_IB_HCA=shca_0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
source ../../../env.sh
CONFIG=${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/oc20/uma_oc20_finetune/uma_sm_finetune_template.yaml
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES torchrun \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --nproc_per_node=8 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$master_addr:$master_port \
  env_lauch.py \
  -c ${CONFIG}
