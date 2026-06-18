#!/bin/bash
#SBATCH --job-name=nep_HfO2_16card
#SBATCH --partition=hpctest01
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_16card_%j.out
#SBATCH --error=slurm_16card_%j.err

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

# 统一走 matchem_env.sh 路径
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
source "$SCRIPT_DIR/../../../../matchem_env.sh"

# MatPL 运行时环境（不侵入 matchem_env.sh，各算例自行维护）
if [ -f "$MATPL_SRC_DIR/env.sh" ]; then
    source "$MATPL_SRC_DIR/env.sh"
fi
export LD_LIBRARY_PATH="$MATPL_SRC_DIR/src/op/build/lib:${LD_LIBRARY_PATH:-}"

# 多节点通信设置
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
function get_free_port() {
    python -c 'import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
MASTER_PORT=$(get_free_port)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# 16 卡训练（2 节点 × 8 DCU）
cd "$SCRIPT_DIR"
srun MatPL train HfO2_nep_train_16card.json
