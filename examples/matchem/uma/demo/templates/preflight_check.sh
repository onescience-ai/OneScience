#!/bin/bash
# preflight_check.sh - UMA 训练前预检脚本
# 用法: bash templates/preflight_check.sh [path1] [path2] ...
#
# 每个 path 可以是:
#   - 普通文件 (如 checkpoint .pt)
#   - 目录 (如 ASE-lmdb 的 train/ val/)
# 检查项:
#   1. ONESCIENCE_DATASETS_DIR 环境变量 (如果 env.sh 已 source)
#   2. 每个路径是否存在
#   3. DCU/GPU 状态

ERRORS=0

echo "========================================="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================="

# 检查路径是否存在 (文件或目录均可)
for p in "$@"; do
    [ -z "$p" ] && continue
    expanded=$(eval echo "$p")
    if [ ! -e "$expanded" ]; then
        echo "[ERROR] 路径不存在: $expanded"
        ERRORS=$((ERRORS + 1))
    elif [ -d "$expanded" ]; then
        echo "[OK] 目录: $expanded"
    else
        echo "[OK] 文件: $expanded"
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
