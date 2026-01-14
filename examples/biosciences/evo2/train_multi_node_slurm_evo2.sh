#!/bin/bash
#SBATCH -J evo2_for_onescience
#SBATCH -p largedev
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=dcu:8
#SBATCH -o evo2_%j.out       
# set -euxo pipefail

module purge
module load sghpc-mpi-gcc/25.8

source /public/home/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
# conda init bash
conda activate onescience-evo2

which python
which hipcc

echo "START TIME: $(date)" 

DEVICES=${SLURM_GPUS_PER_NODE:-8}
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" 


nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

echo "Nodes: ${nodes_array[*]}"

master_addr=${nodes_array[0]}
master_port=29500

export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$((SLURM_NNODES * 8))

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# 建议加上这个，防止多线程库竞争
export OMP_NUM_THREADS=1

srun train_evo2.sh
