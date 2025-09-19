#!/bin/bash
#SBATCH -J md
#SBATCH -p k100ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --output=mul.log
#SBATCH --time=12:00:00

module purge
module load compiler/dtk/25.04

# conda
source /work/home/onescience2025/anaconda3/bin/activate
conda activate macetest

# 环境变量
export HIP_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
# 依据集群情况选一个栈：RCCL(ROCm) 或 NCCL(若在CUDA卡)
export RCCL_IB_HCA=mlx5_0
export RCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# 可选：打印可见设备数做 sanity check
python - <<'PY'
import torch
print("Visible devices:", torch.cuda.device_count())
PY


torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  run_train.py \
  --name='mace01' \
  --model='MACE' \
  --model_dir="./MACE_models" \
  --num_channels=32 \
  --max_L=0 \
  --r_max=4.0 \
  --train_file='./data/solvent_xtb_train_200.xyz' \
  --valid_fraction=0.10 \
  --test_file='./data/solvent_xtb_test.xyz' \
  --energy_key='energy_xtb' \
  --forces_key='forces_xtb' \
  --batch_size=10 \
  --max_num_epochs=100 \
  --swa \
  --seed=123 \
  --distributed \
  --device='cuda' \
  --num_workers=8