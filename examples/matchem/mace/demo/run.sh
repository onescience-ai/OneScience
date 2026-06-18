#!/bin/bash
# run.sh - MACE 训练统一入口脚本
#
# 用法:
#   bash run.sh --config configs/DMC.yaml              # 直接运行训练
#   bash run.sh --config configs/DMC.yaml --submit      # 提交 SLURM 作业
#   bash run.sh --config configs/DMC.yaml --dry-run     # 仅打印命令，不执行
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
# 解析配置
# ============================================================
EXP_NAME=$(python3 "$PARSE_PY" "$CONFIG_ABS" name)
TRAIN_CMD=$(python3 "$PARSE_PY" "$CONFIG_ABS" command)
ENV_EXPORTS=$(python3 "$PARSE_PY" "$CONFIG_ABS" env)
ENV_ARGS=$(python3 "$PARSE_PY" "$CONFIG_ABS" env-args)
DATA_FILES=$(python3 "$PARSE_PY" "$CONFIG_ABS" data-files)

# ============================================================
# Dry-run 模式
# ============================================================
if $DRY_RUN; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$DEMO_DIR/outputs/${EXP_NAME}_${TIMESTAMP}"
    echo "========================================="
    echo "Dry-run: $EXP_NAME"
    echo "Config:  $CONFIG_ABS"
    echo "Output:  $OUTPUT_DIR (未创建)"
    echo "========================================="
    echo ""
    echo "# 环境变量:"
    echo "$ENV_EXPORTS"
    echo ""
    echo "# 训练命令:"
    echo "$TRAIN_CMD"
    echo ""
    echo "# env_setup.sh 参数: $ENV_ARGS"
    echo "# 数据文件:"
    echo "$DATA_FILES" | sed 's/^/#   /'
    exit 0
fi

# ============================================================
# 创建输出目录
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$DEMO_DIR/outputs/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG_ABS" "$OUTPUT_DIR/config.yaml"

# ============================================================
# SLURM 提交模式
# ============================================================
if $SUBMIT; then
    SLURM_VARS=$(python3 "$PARSE_PY" "$CONFIG_ABS" slurm)
    eval "$SLURM_VARS"

    # 检查是否是多节点
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

    # 添加环境初始化
    cat >> "$SLURM_SCRIPT" << 'SETUP_BLOCK'

# 环境初始化
SETUP_BLOCK
    echo "set +u
source /public/home/easyscience2024/wangrui/onescience/examples/matchem/matchem_env.sh
set -u" >> "$SLURM_SCRIPT"

    # 添加预检
    echo "" >> "$SLURM_SCRIPT"
    echo "# 预检" >> "$SLURM_SCRIPT"
    # 将数据文件列表转为参数
    DATA_ARGS=""
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        DATA_ARGS="$DATA_ARGS \"$line\""
    done <<< "$DATA_FILES"
    echo "bash $DEMO_DIR/templates/preflight_check.sh $DATA_ARGS" >> "$SLURM_SCRIPT"

    # 添加环境变量和训练命令
    echo "" >> "$SLURM_SCRIPT"
    echo "# 工作目录" >> "$SLURM_SCRIPT"
    echo "cd $OUTPUT_DIR" >> "$SLURM_SCRIPT"
    echo "" >> "$SLURM_SCRIPT"
    echo "# 环境变量" >> "$SLURM_SCRIPT"
    echo "$ENV_EXPORTS" >> "$SLURM_SCRIPT"
    echo "" >> "$SLURM_SCRIPT"
    echo "# 屏蔽 e3nn FutureWarning 和 TorchScript UserWarning" >> "$SLURM_SCRIPT"
    echo 'export PYTHONWARNINGS="ignore::FutureWarning:e3nn.o3._wigner,ignore::UserWarning:torch.jit._check"' >> "$SLURM_SCRIPT"
    echo "" >> "$SLURM_SCRIPT"
    echo '# 屏蔽 PyTorch NCCL C++ INFO 日志' >> "$SLURM_SCRIPT"
    echo 'export TORCH_CPP_LOG_LEVEL=WARNING' >> "$SLURM_SCRIPT"
    echo 'export NCCL_DEBUG=ERROR' >> "$SLURM_SCRIPT"
    echo '' >> "$SLURM_SCRIPT"
    echo '# 屏蔽 glog INFO（ProcessGroupNCCL.cpp 初始化信息）' >> "$SLURM_SCRIPT"
    echo 'export GLOG_minloglevel=1' >> "$SLURM_SCRIPT"
    echo '' >> "$SLURM_SCRIPT"
    echo '# AMD DCU: 避免 RCCL "Missing HSA_FORCE_FINE_GRAIN_PCIE" 警告' >> "$SLURM_SCRIPT"
    echo 'export HSA_FORCE_FINE_GRAIN_PCIE=1' >> "$SLURM_SCRIPT"

    # 多节点特殊处理
    if [ "$NODES" -gt 1 ]; then
        cat >> "$SLURM_SCRIPT" << 'MULTI_NODE'

# 多节点分布式设置
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29517
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"

# 使用 srun 启动分布式训练
MULTI_NODE
        # srun 包裹训练命令
        echo "srun --export=ALL bash -c '" >> "$SLURM_SCRIPT"
        echo "  export RANK=\$SLURM_PROCID" >> "$SLURM_SCRIPT"
        echo "  export LOCAL_RANK=\$SLURM_LOCALID" >> "$SLURM_SCRIPT"
        echo "  exec $TRAIN_CMD" >> "$SLURM_SCRIPT"
        echo "'" >> "$SLURM_SCRIPT"
    else
        echo "" >> "$SLURM_SCRIPT"
        echo "# 训练命令" >> "$SLURM_SCRIPT"
        echo "$TRAIN_CMD" >> "$SLURM_SCRIPT"
    fi

    echo "========================================="
    echo "SLURM 脚本已生成: $SLURM_SCRIPT"
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
echo "========================================="

# 环境初始化
set +u
source /public/home/easyscience2024/wangrui/onescience/examples/matchem/matchem_env.sh
set -u

# 预检
DATA_ARGS=""
while IFS= read -r line; do
    [ -z "$line" ] && continue
    DATA_ARGS="$DATA_ARGS \"$line\""
done <<< "$DATA_FILES"
eval "bash $DEMO_DIR/templates/preflight_check.sh $DATA_ARGS"

# 设置环境变量
eval "$ENV_EXPORTS"

# 屏蔽 e3nn FutureWarning 和 TorchScript UserWarning
export PYTHONWARNINGS="ignore::FutureWarning:e3nn.o3._wigner,ignore::UserWarning:torch.jit._check"

# 屏蔽 PyTorch NCCL C++ INFO 日志
export TORCH_CPP_LOG_LEVEL=WARNING
export NCCL_DEBUG=ERROR

# 屏蔽 glog INFO（ProcessGroupNCCL.cpp 初始化信息）
export GLOG_minloglevel=1

# AMD DCU: 避免 RCCL "Missing HSA_FORCE_FINE_GRAIN_PCIE" 警告
export HSA_FORCE_FINE_GRAIN_PCIE=1

# 切换到输出目录
cd "$OUTPUT_DIR"

# 执行训练
echo "========================================="
echo "开始训练..."
echo "========================================="
if [ -n "${TRAIN_CMD:-}" ]; then
    eval "$TRAIN_CMD"
else
    echo "[FATAL] TRAIN_CMD 为空，无法执行任务"
    exit 1
fi
