#!/bin/bash
# run.sh - UMA 训练统一入口脚本
#
# 用法:
#   bash run.sh --config configs/oc20_ef_4dcu.yaml              # 直接运行训练
#   bash run.sh --config configs/oc20_ef_4dcu.yaml --submit     # 提交 SLURM 作业
#   bash run.sh --config configs/oc20_ef_4dcu.yaml --dry-run    # 仅打印命令，不执行
#
set -euo pipefail

# ============================================================
# 参数解析
# ============================================================
CONFIG=""
SUBMIT=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --config=*)
            CONFIG="${1#*=}"; shift ;;
        --submit)
            SUBMIT=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            echo "用法: bash run.sh --config <config.yaml> [--submit] [--dry-run]"
            echo ""
            echo "选项:"
            echo "  --config <file>   YAML 配置文件路径 (必需)"
            echo "  --submit          生成 SLURM 脚本并提交作业"
            echo "  --dry-run         仅打印训练命令，不执行"
            exit 0 ;;
        *)
            echo "[ERROR] 未知参数: $1"
            exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "[ERROR] 请指定配置文件: bash run.sh --config configs/xxx.yaml"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] 配置文件不存在: $CONFIG"
    exit 1
fi

# ============================================================
# 路径设置
# ============================================================
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
PARSE_PY="$DEMO_DIR/_parse_config.py"

# ============================================================
# 解析 meta 部分 (不依赖 hydra_config 路径)
# ============================================================
EXP_NAME=$(python3 "$PARSE_PY" "$CONFIG_ABS" name)
ENV_EXPORTS=$(python3 "$PARSE_PY" "$CONFIG_ABS" env)
ENV_ARGS=$(python3 "$PARSE_PY" "$CONFIG_ABS" env-args)
DATA_FILES=$(python3 "$PARSE_PY" "$CONFIG_ABS" data-files)

# ============================================================
# 预生成输出目录与 hydra_config.yaml（dry-run 也生成，便于预览）
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$DEMO_DIR/outputs/${EXP_NAME}_${TIMESTAMP}"
HYDRA_CONFIG="$OUTPUT_DIR/hydra_config.yaml"

# dry-run 不实际创建目录
if $DRY_RUN; then
    DRY_HYDRA_TMP=$(mktemp -t "uma_hydra_XXXXXX.yaml")
    HYDRA_CONFIG="$DRY_HYDRA_TMP"
    python3 "$PARSE_PY" "$CONFIG_ABS" hydra-config > "$HYDRA_CONFIG"
else
    mkdir -p "$OUTPUT_DIR"
    cp "$CONFIG_ABS" "$OUTPUT_DIR/config.yaml"
    python3 "$PARSE_PY" "$CONFIG_ABS" hydra-config > "$HYDRA_CONFIG"
    # UMA/calculate/pretrained_mlip.py 使用 os.getcwd()+"/models/pretrained_models.json"
    # 定位预训练模型清单, 而 run.sh 会 cd 到 OUTPUT_DIR 再启动训练, 所以在 OUTPUT_DIR
    # 下放一个指回 UMA/models 的软链, 保证 cwd 相对路径能找到 json。
    UMA_ROOT_DIR="$(cd "$DEMO_DIR/.." && pwd)"
    if [ -d "$UMA_ROOT_DIR/models" ] && [ ! -e "$OUTPUT_DIR/models" ]; then
        ln -s "$UMA_ROOT_DIR/models" "$OUTPUT_DIR/models"
    fi
fi

# 现在 hydra_config 路径就绪, 拼 command
TRAIN_CMD=$(python3 "$PARSE_PY" "$CONFIG_ABS" command "$HYDRA_CONFIG")

# ============================================================
# Dry-run 模式
# ============================================================
if $DRY_RUN; then
    echo "========================================="
    echo "Dry-run: $EXP_NAME"
    echo "Config:  $CONFIG_ABS"
    echo "Output:  $OUTPUT_DIR (未创建)"
    echo "Hydra:   $HYDRA_CONFIG (临时)"
    echo "========================================="
    echo ""
    echo "# 环境变量:"
    echo "$ENV_EXPORTS"
    echo ""
    echo "# 训练命令:"
    echo "$TRAIN_CMD"
    echo ""
    echo "# env_setup.sh 参数: $ENV_ARGS"
    echo "# 预检路径:"
    echo "$DATA_FILES" | sed 's/^/#   /'
    echo ""
    echo "# hydra_config.yaml 预览 (前 40 行):"
    head -n 40 "$HYDRA_CONFIG" | sed 's/^/#   /'
    rm -f "$HYDRA_CONFIG"
    exit 0
fi

# ============================================================
# SLURM 提交模式
# ============================================================
if $SUBMIT; then
    SLURM_VARS=$(python3 "$PARSE_PY" "$CONFIG_ABS" slurm)
    eval "$SLURM_VARS"

    SLURM_SCRIPT="$OUTPUT_DIR/submit.sh"

    # 生成 SLURM header
    sed -e "s|{{JOB_NAME}}|${JOB_NAME}|g" \
        -e "s|{{PARTITION}}|${PARTITION}|g" \
        -e "s|{{NODES}}|${NODES}|g" \
        -e "s|{{NTASKS_PER_NODE}}|${NTASKS_PER_NODE}|g" \
        -e "s|{{CPUS_PER_TASK}}|${CPUS_PER_TASK}|g" \
        -e "s|{{GPUS_PER_NODE}}|${GPUS_PER_NODE}|g" \
        -e "s|{{TIME}}|${TIME}|g" \
        "$DEMO_DIR/templates/slurm_header.template" > "$SLURM_SCRIPT"

    # 环境初始化
    cat >> "$SLURM_SCRIPT" << 'SETUP_BLOCK'

# 环境初始化
SETUP_BLOCK
    echo "source $DEMO_DIR/templates/env_setup.sh $ENV_ARGS" >> "$SLURM_SCRIPT"

    # 预检
    echo "" >> "$SLURM_SCRIPT"
    echo "# 预检" >> "$SLURM_SCRIPT"
    DATA_ARGS=""
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        DATA_ARGS="$DATA_ARGS \"$line\""
    done <<< "$DATA_FILES"
    echo "bash $DEMO_DIR/templates/preflight_check.sh $DATA_ARGS" >> "$SLURM_SCRIPT"

    # 工作目录与环境变量
    echo "" >> "$SLURM_SCRIPT"
    echo "# 工作目录" >> "$SLURM_SCRIPT"
    echo "cd $OUTPUT_DIR" >> "$SLURM_SCRIPT"
    echo "" >> "$SLURM_SCRIPT"
    echo "# 环境变量" >> "$SLURM_SCRIPT"
    echo "$ENV_EXPORTS" >> "$SLURM_SCRIPT"
    echo "" >> "$SLURM_SCRIPT"
    echo "# 屏蔽 PyTorch NCCL C++ INFO 日志" >> "$SLURM_SCRIPT"
    echo "export TORCH_CPP_LOG_LEVEL=WARNING" >> "$SLURM_SCRIPT"
    echo "export NCCL_DEBUG=ERROR" >> "$SLURM_SCRIPT"
    echo "export GLOG_minloglevel=1" >> "$SLURM_SCRIPT"
    echo "" >> "$SLURM_SCRIPT"
    echo "# AMD DCU: 避免 RCCL \"Missing HSA_FORCE_FINE_GRAIN_PCIE\" 警告" >> "$SLURM_SCRIPT"
    echo "export HSA_FORCE_FINE_GRAIN_PCIE=1" >> "$SLURM_SCRIPT"
    echo "" >> "$SLURM_SCRIPT"
    echo "# UMA 依赖: 清理 rocblas tensile 路径, 避免拉到错版本" >> "$SLURM_SCRIPT"
    echo "unset ROCBLAS_TENSILE_LIBPATH" >> "$SLURM_SCRIPT"

    # 多节点特殊处理: 设置 MASTER_ADDR/PORT, 用 srun 包裹
    if [ "$NODES" -gt 1 ]; then
        cat >> "$SLURM_SCRIPT" << 'MULTI_NODE'

# 多节点分布式设置
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
export MASTER_ADDR=${nodes_array[0]}
export MASTER_PORT=29504
echo "SLURM_NNODES=$SLURM_NNODES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# srun 启动分布式训练 (每节点 1 个 task, torchrun 在节点内 spawn 多个 rank)
MULTI_NODE
        echo "srun --nodes=\$SLURM_NNODES --ntasks=\$SLURM_NNODES \\" >> "$SLURM_SCRIPT"
        # 把 TRAIN_CMD 原样追加 (其中的 \${SLURM_*} / \${MASTER_*} 会在每节点上展开)
        echo "  $TRAIN_CMD" >> "$SLURM_SCRIPT"
    else
        echo "" >> "$SLURM_SCRIPT"
        echo "# 训练命令" >> "$SLURM_SCRIPT"
        echo "$TRAIN_CMD" >> "$SLURM_SCRIPT"
    fi

    echo "========================================="
    echo "SLURM 脚本已生成: $SLURM_SCRIPT"
    echo "hydra 配置已生成: $HYDRA_CONFIG"
    echo "配置快照已保存: $OUTPUT_DIR/config.yaml"
    echo "========================================="
    echo ""
    echo "提交作业..."
    sbatch "$SLURM_SCRIPT"
    exit 0
fi

# ============================================================
# 直接运行模式
# ============================================================
echo "========================================="
echo "实验: $EXP_NAME"
echo "配置: $CONFIG_ABS"
echo "输出: $OUTPUT_DIR"
echo "Hydra: $HYDRA_CONFIG"
echo "========================================="

# 环境初始化
# shellcheck disable=SC2086
source "$DEMO_DIR/templates/env_setup.sh" $ENV_ARGS

# 预检
DATA_ARGS=""
while IFS= read -r line; do
    [ -z "$line" ] && continue
    DATA_ARGS="$DATA_ARGS \"$line\""
done <<< "$DATA_FILES"
eval "bash $DEMO_DIR/templates/preflight_check.sh $DATA_ARGS"

# 设置环境变量
eval "$ENV_EXPORTS"

# 屏蔽 PyTorch NCCL C++ INFO 日志
export TORCH_CPP_LOG_LEVEL=WARNING
export NCCL_DEBUG=ERROR
export GLOG_minloglevel=1

# AMD DCU
export HSA_FORCE_FINE_GRAIN_PCIE=1
unset ROCBLAS_TENSILE_LIBPATH

# 切换到输出目录
cd "$OUTPUT_DIR"

# 执行训练
# 执行训练（实时合并日志）
echo "========================================="
echo "开始训练..."
echo "========================================="

export PYTHONUNBUFFERED=1
MERGED_LOG="${OUTPUT_DIR}/train_merged.out"
: > "${MERGED_LOG}"
echo "[LOG] merged realtime log: ${MERGED_LOG}"

set +e
eval "$TRAIN_CMD" 2>&1 | tee -a "${MERGED_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

exit "${TRAIN_EXIT}"

