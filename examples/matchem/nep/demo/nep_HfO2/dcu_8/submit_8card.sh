#!/bin/bash
#SBATCH --job-name=nep_HfO2_8card
#SBATCH --partition=hpctest01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_8card_%j.out
#SBATCH --error=slurm_8card_%j.err

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

# 统一走 matchem_env.sh 路径
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
source "$SCRIPT_DIR/../../../../matchem_env.sh"

# MatPL 运行时环境（不侵入 matchem_env.sh，各算例自行维护）
if [ -f "$MATPL_SRC_DIR/env.sh" ]; then
    source "$MATPL_SRC_DIR/env.sh"
fi
export LD_LIBRARY_PATH="$MATPL_SRC_DIR/src/op/build/lib:${LD_LIBRARY_PATH:-}"

# 8 卡训练
cd "$SCRIPT_DIR"
MatPL train HfO2_nep_train_8card.json
