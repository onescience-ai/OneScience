#!/bin/bash
# ==========================================
# MatChem 统一环境配置脚本
# ==========================================

# ---------- 1. 基础环境配置 ----------
export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-matchem_pip}"

# 获取本脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR
# ---------- 2. OneScience 运行时环境变量 ----------
source "$SCRIPT_DIR/../../env.sh"

# ---------- 3. 外部软件路径 ----------
export DEEPMD_SRC_DIR="${DEEPMD_SRC_DIR:-/path/to/deepmd-kit}"
export MATPL_SRC_DIR="${MATPL_SRC_DIR:-/path/to/matpl_dcu}"
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-/path/to/lammps_dcu}"
export DP_CPP_DIR="${DP_CPP_DIR:-/path/to/dp_cpp_dcu}"


# ---------- 5. 加载集群模块与 conda（set +u 状态）----------
source ~/.bashrc

module load sghpcdas/25.6        || { echo "❌ 加载模块 sghpcdas/25.6 失败"; return 1; }
module load sghpc-mpi-gcc/26.3   || { echo "❌ 加载模块 sghpc-mpi-gcc/26.3 失败"; return 1; }

# conda 激活必须在 set +u 状态下执行！
if ! conda env list 2>/dev/null | grep -q "^${MATCHEM_CONDA_NAME} "; then
    echo "❌ Conda 环境 '${MATCHEM_CONDA_NAME}' 不存在！" >&2
    echo "   可用环境：" >&2
    conda env list >&2
    return 1
fi

conda activate "$MATCHEM_CONDA_NAME" || { echo "❌ 激活 conda 环境失败"; return 1; }

# 验证激活成功
if [[ -z "${CONDA_PREFIX:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$MATCHEM_CONDA_NAME" ]]; then
    echo "❌ Conda 环境激活验证失败" >&2
    return 1
fi

# ---------- 6. LAMMPS 运行时环境 ----------
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib_override:${LD_LIBRARY_PATH:-}
export LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib

echo "✅ MatChem 环境已激活: ${MATCHEM_CONDA_NAME}"
echo "   📁 Datasets: ${ONESCIENCE_DATASETS_DIR}"
echo "   📁 Models:   ${ONESCIENCE_MODELS_DIR}"
echo "   🐍 Python:   $(which python 2>/dev/null || echo '未找到')"
echo "   🏠 Conda:    ${CONDA_PREFIX:-}"
