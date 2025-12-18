#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -N 8
#SBATCH -J train
#SBATCH -t 480:00:00
#SBATCH -p kshdexclu09
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --mem=110G

export MASTER_ADDR=$(hostname)
export NCCL_IB_HCA=mlx5_0:1

set -x
srun -u --mpi=pmix_v3 \
    bash -c "
    source export_DDP_vars.sh
    bash ./$1
    "

