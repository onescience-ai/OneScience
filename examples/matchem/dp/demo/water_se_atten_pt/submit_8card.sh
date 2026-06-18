#!/bin/bash
#SBATCH --job-name=dp_pt_atten_8card
#SBATCH --partition=hpctest01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_8card_%j.out
#SBATCH --error=slurm_8card_%j.err

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

# 统一走 matchem_env.sh 路径
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
source "$SCRIPT_DIR/../../../matchem_env.sh"

# DeepMD 训练环境已由 matchem_env.sh 覆盖

# 多卡训练（8卡）
cd "$SCRIPT_DIR"
torchrun --nproc_per_node=8 -m deepmd --pt train input_torch.json
