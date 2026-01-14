#!/bin/bash
#SBATCH -p large_ai4s
#SBATCH -x b09r2n03,b09r1n09
#SBATCH -N 8
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=6
#SBATCH -J Xihe_16_6
#SBATCH --time=72:00:00
#SBATCH -o logs/%x.out
#SBATCH --exclusive

echo "START TIME: $(date)"
module purge
##### Launch Conda #####
module load sghpcdas/25.6 
conda init bash
source ~/.bashrc
##### Activate Conda env #####
conda activate xiheh
##### Launch DTK #####
module load sghpc-mpi-gcc/25.8
##### Show env #####
which python
which hipcc
####Launch env ####
source  ../../../env.sh

##### Set DCU #####
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OMP_NUM_THREADS=16
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# 第一个节点的地址
export MASTER_ADDR=$(hostname)

# 在每个节点上启动 torchrun
echo SLURM_NNODES=$SLURM_NNODES
echo "Nodes: ${nodes_array[*]}"
echo SLURM_NTASKS=$SLURM_NTASKS

srun -u --mpi=pmix\
    bash -c "
    source export_DDP_vars.sh
    python train.py
    "
