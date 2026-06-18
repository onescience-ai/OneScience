#!/bin/bash
# preflight_check.sh - 训练前预检脚本
# 用法: bash templates/preflight_check.sh [data_file1] [data_file2] ...
#
# 检查项:
#   1. ONESCIENCE_DATASETS_DIR 环境变量
#   2. 数据文件是否存在
#   3. DCU/GPU 状态
#   4. 打印节点信息

ERRORS=0

echo "========================================="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================="

# 检查数据集目录环境变量
if [ -z "${ONESCIENCE_DATASETS_DIR:-}" ]; then
    echo "[ERROR] ONESCIENCE_DATASETS_DIR 未设置，请先 source env.sh"
    ERRORS=$((ERRORS + 1))
fi

# 检查数据文件是否存在
for f in "$@"; do
    # 跳过空参数
    [ -z "$f" ] && continue
    # 展开环境变量后检查
    expanded=$(eval echo "$f")
    if [ ! -e "$expanded" ]; then
        echo "[ERROR] 数据文件不存在: $expanded"
        ERRORS=$((ERRORS + 1))
    else
        echo "[OK] 数据文件: $expanded"
    fi
done

# 打印 DCU/GPU 状态
echo "-----------------------------------------"
echo ">>> DCU/GPU 状态:"
if command -v hy-smi &>/dev/null; then
    hy-smi
elif command -v rocm-smi &>/dev/null; then
    rocm-smi
elif command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "[WARN] 未找到 GPU 监控工具 (hy-smi/rocm-smi/nvidia-smi)"
fi
echo "-----------------------------------------"

if [ $ERRORS -gt 0 ]; then
    echo "[FATAL] 预检发现 $ERRORS 个错误，终止训练"
    exit 1
fi

echo "[OK] 预检通过"
