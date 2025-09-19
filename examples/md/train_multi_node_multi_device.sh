#!/bin/bash
#SBATCH -J mace
#SBATCH -p k100ai
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --time=12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.out

module purge
module load compiler/dtk/25.04
source /work/home/onescience2025/anaconda3/bin/activate
conda activate macetest

# 必要环境
export OMP_NUM_THREADS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export NCCL_PROTO=Simple

# 分布式 env://
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29517
export WORLD_SIZE=$SLURM_NTASKS

# 路径
TRAIN_FILE=./data/solvent_xtb_train_200.xyz
TEST_FILE=./data/solvent_xtb_test.xyz

# 启动 8 个 rank
srun --export=ALL --chdir="$WORKDIR" bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE='"$WORLD_SIZE"'
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'

  exec python run_train.py \
    --name mace01 \
    --model MACE \
    --model_dir="./MACE_models" \
    --num_channels 32 \
    --max_L 0 \
    --r_max 4.0 \
    --train_file '"$TRAIN_FILE"' \
    --valid_fraction 0.10 \
    --test_file '"$TEST_FILE"' \
    --energy_key energy_xtb \
    --forces_key forces_xtb \
    --batch_size 10 \
    --max_num_epochs 100 \
    --swa \
    --seed 123 \
    --distributed \
    --device cuda \
    --num_workers 8
'
