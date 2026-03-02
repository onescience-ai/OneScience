#!/bin/bash
#SBATCH -J md_multi_node
#SBATCH -p newlarge
#SBATCH --nodes=2              # 多节点训练
#SBATCH --ntasks-per-node=8    # 使用 srun 启动时，这里的任务数必须等于单节点的显卡数
#SBATCH --cpus-per-task=8      # 调整为每个进程8个CPU 核心(8进程x8核心 = 单节点共用64核心)
#SBATCH --gres=dcu:8           # 申请单节点8张DCU
#SBATCH --output=%j.log
#SBATCH --time=8:00:00

module purge
source ~/.bashrc
conda activate onescience_test
module load compiler/dtk/25.04.2

echo "========================================="
echo "Nodes allocated: $SLURM_JOB_NODELIST"
echo "Time: $(date)"
echo "========================================="

# 在主节点上进行训练前的显存检查
echo ">>> Pre-training DCU Status (Checking RAM on Master):"
hy-smi
echo "-----------------------------------------"

# DCU/ROCm 多节点通信必需的环境变量
export OMP_NUM_THREADS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
# 注意：你需要根据你们超算的实际硬件，确认 InfiniBand 网卡的名字是不是 ib0 和 mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export NCCL_PROTO=Simple

# 设置分布式的 env:// 环境变量
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29517
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"

# 使用 srun 启动分布式训练
# srun 会自动在你申请的所有节点和显卡上拉起对应数量的进程
srun --export=ALL bash -c '
  # Slurm 会自动给每个任务分配 ID，我们将它们转为 PyTorch 需要的环境变量
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  

  exec python ../../train.py \
    --name="nanotube_large_r55_l_2" \
    --train_file="../../data/nanotube/nanotube_large.xyz" \
    --valid_fraction=0.05 \
    --test_file="../../data/nanotube/nanotube_test.xyz" \
    --E0s="average" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=256 \
    --max_L=2 \
    --correlation=3 \
    --r_max=5.0 \
    --forces_weight=1000 \
    --energy_weight=10 \
    --energy_key="Energy" \
    --forces_key="forces" \
    --batch_size=2 \
    --valid_batch_size=4 \
    --max_num_epochs=100 \
    --start_swa=60 \
    --scheduler_patience=5 \
    --patience=15 \
    --eval_interval=10 \
    --ema \
    --swa \
    --swa_forces_weight=10 \
    --error_table="PerAtomMAE" \
    --default_dtype="float64" \
    --device=cuda \
    --distributed \
    --seed=123 \
    --restart_latest \
    --save_cp
'
