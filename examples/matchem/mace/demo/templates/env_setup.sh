#!/bin/bash
# env_setup.sh - 公共环境初始化脚本
# 用法: source templates/env_setup.sh <conda_env> <module1> [module2 ...]
#
# 参数:
#   $1        - conda 环境名 (如 chem)
#   $2 ...    - 需要加载的 module 列表

CONDA_ENV="${1:?请指定 conda 环境名}"
shift

# 系统脚本 (bashrc, conda, module) 可能引用未初始化变量或返回非零退出码，
# 临时关闭 errexit 和 nounset 避免报错
set +eu

# 清理环境
module purge
source ~/.bashrc

# 激活 conda 环境
conda activate "$CONDA_ENV"


# 加载指定 modules
for mod in "$@"; do
    module load "$mod"
done

# 恢复 errexit 和 nounset
set -eu

# 加载数据集路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$DEMO_DIR/../../../.." && pwd)"
source "$REPO_ROOT/env.sh"
