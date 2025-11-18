#!/bin/bash
#SBATCH -p largedev
#SBATCH -N 4
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J GraphCast
#SBATCH -o logs/%j.out
#SBATCH --exclusive

echo "START TIME: $(date)"
module purge
##### Launch Conda #####
module load sghpcdas/25.6 
conda init bash
source ~/.bashrc
##### Activate Conda env #####
conda activate era5newdata
##### Launch DTK #####
module load sghpc-mpi-gcc/25.8
##### Show env #####
which python
which hipcc
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