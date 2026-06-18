#!/bin/bash
# ==========================================
# OneScience MatChem 环境一键安装脚本
# 用途：创建 conda 环境并安装 OneScience[matchem] 及其 DCU 依赖
# 用法：bash matchem_install.sh
# ==========================================
set -euo pipefail

# ---------- 1. 配置区（可按需修改） ----------
MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-matchem_opt}"
ONESCIENCE_MAIN_DIR="${ONESCIENCE_MAIN_DIR:-/public/home/easyscience2024/wangrui/onescience}"

# ---------- 2. 加载基础模块与环境 ----------
# 先关闭 set -u，避免 /etc/bashrc 中未绑定变量报错
set +u
source ~/.bashrc
set -u
module load sghpcdas/25.6
module load sghpc-mpi-gcc/26.3

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到 conda，请确认已正确加载 sghpcdas 模块。"
    exit 1
fi

# ---------- 3. 创建 conda 环境 ----------
# 若环境已存在则跳过，避免误覆盖
if conda env list | grep -qE "^${MATCHEM_CONDA_NAME}\s"; then
    echo "[提示] conda 环境 '${MATCHEM_CONDA_NAME}' 已存在，跳过创建。"
    echo "[提示] 如需重建，请先执行：conda remove -n ${MATCHEM_CONDA_NAME} --all -y"
else
    echo "[步骤 1/4] 创建 conda 环境：${MATCHEM_CONDA_NAME} ..."
    conda create -n "${MATCHEM_CONDA_NAME}" python=3.11 -y
fi

# ---------- 4. 安装 uv 工具 ----------
echo "[步骤 2/4] 激活环境并安装 uv ..."
conda activate "${MATCHEM_CONDA_NAME}"
python -m pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# ---------- 5. 安装 OneScience[matchem] ----------
echo "[步骤 3/4] 安装 OneScience[matchem] ..."
cd "${ONESCIENCE_MAIN_DIR}"
bash install.sh matchem

# ---------- 6. 安装验证 ----------
echo "[步骤 4/4] 验证安装结果 ..."
python -c "import torch; import onescience; print('torch 版本:', torch.__version__); print('onescience: 导入成功')"

echo ""
echo "============================================"
echo "  MatChem 环境安装完成"
echo "  环境名：${MATCHEM_CONDA_NAME}"
echo "  激活命令：conda activate ${MATCHEM_CONDA_NAME}"
echo "============================================"
