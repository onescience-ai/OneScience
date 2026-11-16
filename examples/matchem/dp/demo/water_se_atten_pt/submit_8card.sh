#!/bin/bash
#SBATCH --job-name=dp_pt_atten_8card
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_8card_%j.out
#SBATCH --error=slurm_8card_%j.err

if [[ -n "${SCRIPT_DIR}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    echo "ERROR: 未检测到外部 SCRIPT_DIR 变量，也不在 Slurm 任务环境（SLURM_SUBMIT_DIR 为空）"
    exit 1
fi
echo $SCRIPT_DIR

export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-test_pip}"
source "$SCRIPT_DIR/matchem_env.sh"

# DeepMD 训练环境已由 matchem_env.sh 覆盖

# 多卡训练（8卡）

# 输入 JSON 使用环境变量，运行前展开为临时文件
INPUT_JSON="input_torch.json"
EXPANDED_JSON=".${INPUT_JSON%.json}_expanded_$$.json"
trap 'rm -f "$EXPANDED_JSON"' EXIT
python3 -c "import os; open('$EXPANDED_JSON','w').write(os.path.expandvars(open('$INPUT_JSON').read()))"

torchrun --nproc_per_node=8 -m deepmd --pt train "$EXPANDED_JSON"
