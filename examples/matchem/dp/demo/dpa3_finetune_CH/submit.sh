#!/bin/bash
#SBATCH --job-name=dpa3_finetune
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

if [[ -n "${SCRIPT_DIR}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    echo "ERROR: 未检测到外部 SCRIPT_DIR 变量，也不在 Slurm 任务环境（SLURM_SUBMIT_DIR 为空）"
    exit 1
fi
export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-test_pip}"
source "${SCRIPT_DIR}/matchem_env.sh"
echo $ONESCIENCE_DATASETS_DIR
# 限制并行度，规避 ROCm kernel launch 问题
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1
export OMP_NUM_THREADS=1

# 输入 JSON 使用环境变量，运行前展开为临时文件
INPUT_JSON="input_finetune.json"
EXPANDED_JSON=".${INPUT_JSON%.json}_expanded_$$.json"
trap 'rm -f "$EXPANDED_JSON"' EXIT
python3 -c "import os; open('$EXPANDED_JSON','w').write(os.path.expandvars(open('$INPUT_JSON').read()))"

dp --pt train "$EXPANDED_JSON" --finetune ./DPA-3.1-3M.pt
