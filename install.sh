#!/bin/bash

# 开启严格模式：遇到错误、未定义变量或管道错误时立即退出
set -euo pipefail
pip install uv
# ==========================================
# 1. 参数校验与环境准备
# ==========================================
DOMAIN="${1:-all}" # 获取用户输入的参数，默认安装全部领域

# 处理帮助命令
if [[ "$DOMAIN" == "-h" || "$DOMAIN" == "--help" ]]; then
    echo "Usage: bash $0 [earth|cfd|bio|matchem|all]"
    echo "If no domain is specified, all domains will be installed by default."
    exit 0
fi

# 校验输入的领域是否支持
case "$DOMAIN" in
    earth|cfd|bio|matchem|all)
        # 校验通过，继续执行
        ;;
    *)
        echo "Error: Unsupported domain '$DOMAIN'"
        echo "Usage: bash $0 [earth|cfd|bio|matchem|all]"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS_FILE="$SCRIPT_DIR/constraints.txt"

# ==========================================
# 2. 依赖配置区 (后续维护仅需修改此处)
# ==========================================
# 约定：
# - CORE_WHEELS: 所有领域都依赖的基础轮子
# - EARTH/CFD/BIO/MATCHEM_WHEELS: 特定领域的定制轮子

CORE_WHEELS=(
    # 基础依赖占位：如果所有领域都需要某个轮子，添加在这里
)

EARTH_WHEELS=(
    "https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.7/torch-2.5.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/vision/DAS1.7/torchvision-0.20.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/apex/DAS1.6/apex-1.4.0+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/onnxruntime/DAS1.7/onnxruntime-1.19.2+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/dgl/DAS1.7/dgl-2.2.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
)

CFD_WHEELS=(
    "https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.7/torch-2.5.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/vision/DAS1.7/torchvision-0.20.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/fastpt/DAS1.7/fastpt-2.1.1+das.dtk25042-cp311-cp311-linux_x86_64.whl"
    # "https://download.sourcefind.cn:65024/directlink/4/fastpt/DAS1.7/fastpt-2.2.0+das.dtk25042-cp311-cp311-linux_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/dgl/DAS1.7/dgl-2.2.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/tensorflow/DAS1.7/tensorflow-2.18.0+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/9/onesicence/dtk-25.04.2/torch_scatter-2.1.0+das.opt1.dtk25043-cp311-cp311-linux_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/9/onesicence/dtk-25.04.2/torch_cluster-1.6.3+das.opt1.dtk25043-cp311-cp311-linux_x86_64.whl"
)

BIO_WHEELS=(
    "https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.7/torch-2.5.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/vision/DAS1.7/torchvision-0.20.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/jax/DAS1.7/jax-0.4.34+das.opt1.dtk25042-py3-none-any.whl"
    "https://download.sourcefind.cn:65024/file/4/jax/DAS1.7/jax_rocm60_pjrt-0.4.34+das.opt1.dtk25042-py3-none-manylinux2014_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/jax/DAS1.7/jax_rocm60_plugin-0.4.34+das.opt1.dtk25042-cp311-cp311-manylinux2014_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/jax/DAS1.7/jaxlib-0.4.34+das.opt1.dtk25042-cp311-cp311-manylinux2014_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/transformer_engine/DAS1.7/transformer_engine-2.5.0+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/flash_attn/DAS1.7/flash_attn-2.6.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/9/onesicence/dtk-25.04.2/bionemo_noodles-0.1.2-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/9/onesicence/dtk-25.04.2/nemo_toolkit-2.5.0rc0-py3-none-any.whl"
    "https://download.sourcefind.cn:65024/directlink/9/onesicence/dtk-25.04.2/openmm-8.3.1-cp311-cp311-linux_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/triton/DAS1.7/triton-3.1.0+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/jax_triton/DAS1.7/jax_triton-0.2.0+das.opt1.dtk25042-py3-none-any.whl"
    "https://download.sourcefind.cn:65024/file/4/tensorflow/DAS1.7/tensorflow-2.18.0+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/dgl/DAS1.7/dgl-2.2.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    # "https://download.sourcefind.cn:65024/file/4/dgl/DAS1.7/dgl-2.2.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/megatron_core-0.15.0-cp311-cp311-linux_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/causal_conv1d-1.5.2-cp311-cp311-linux_x86_64.whl"
)

MATCHEM_WHEELS=(
    "https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.7/torch-2.5.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/vision/DAS1.7/torchvision-0.20.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/dgl/DAS1.7/dgl-2.2.1+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/directlink/4/fastpt/DAS1.7/fastpt-2.1.1+das.dtk25042-cp311-cp311-linux_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/triton/DAS1.7/triton-3.1.0+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl"
    "https://download.sourcefind.cn:65024/file/4/tensorflow/DAS1.7/tensorflow-2.18.0+das.opt1.dtk25042-cp311-cp311-manylinux_2_28_x86_64.whl" # 跑DP模型会有JIT编译错误，需切换到DAS1.8版本
    #"https://download.sourcefind.cn:65024/file/4/tensorflow/DAS1.8/tensorflow-2.18.0+das.opt1.dtk2604-cp311-cp311-manylinux_2_28_x86_64.whl" 
#torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
    )

# ==========================================
# 3. 核心安装逻辑
# ==========================================

# 对轮子链接按首次出现顺序去重，兼容不支持关联数组的 Bash 版本
dedupe_args() {
    local deduped=()
    local item
    local existing
    local found

    for item in "$@"; do
        found=0
        for existing in "${deduped[@]}"; do
            if [[ "$existing" == "$item" ]]; then
                found=1
                break
            fi
        done

        if [[ "$found" -eq 0 ]]; then
            deduped+=("$item")
        fi
    done

    printf '%s\n' "${deduped[@]}"
}

# 统一的轮子安装函数
install_wheels() {
    local label="$1"
    shift # 移除第一个参数 (label)，剩下的全都是轮子链接
    local deduped_wheels=()

    # 如果没有传入具体的轮子，则跳过
    if [[ "$#" -eq 0 ]]; then
        echo ">>> No $label DAS wheels to install. Skipping."
        return 0
    fi

    while IFS= read -r wheel; do
        [[ -n "$wheel" ]] && deduped_wheels+=("$wheel")
    done < <(dedupe_args "$@")

    echo ">>> Installing ${label} DAS wheels..."
    # 优化：一次性传递所有轮子给 pip，避免 for 循环，提升安装速度并让 pip 更好地处理依赖关系
    uv pip install "${deduped_wheels[@]}" --system
}

# 1. 安装核心基础依赖
install_wheels "core" "${CORE_WHEELS[@]}"

# 2. 根据用户传入的 domain 安装专属依赖
case "$DOMAIN" in
    earth)
        install_wheels "earth" "${EARTH_WHEELS[@]}"
        ;;
    cfd)
        install_wheels "cfd" "${CFD_WHEELS[@]}"      
        ;;
    bio)
        install_wheels "bio" "${BIO_WHEELS[@]}"
        ;;
    matchem)
        install_wheels "matchem" "${MATCHEM_WHEELS[@]}"
        echo ">>> Installing cuequivariance base package"
        uv pip install cuequivariance==0.8.0
        ;;
    all)
        install_wheels "all" \
            "${EARTH_WHEELS[@]}" \
            "${CFD_WHEELS[@]}" \
            "${BIO_WHEELS[@]}" \
            "${MATCHEM_WHEELS[@]}"
        ;;
esac

EXTRAS_SPEC="$DOMAIN"
if [[ "$DOMAIN" == "all" ]]; then
    EXTRAS_SPEC="earth,cfd,bio,matchem"
fi

if [[ "$DOMAIN" != "bio" && "$DOMAIN" != "all" ]]; then
    export ONESCIENCE_SKIP_AF3_BUILD=1
fi
if [[ "$DOMAIN" == "bio" || "$DOMAIN" == "all" ]]; then
    export ALPHAFOLD3_TMP_ROOT="$SCRIPT_DIR/af3_build"
    export ALPHAFOLD3_CIFPP_COMPONENTS=$ALPHAFOLD3_TMP_ROOT/components.cif
    export ALPHAFOLD3_CIFPP_DATA_DIR=$ALPHAFOLD3_TMP_ROOT
    mkdir -p "$ALPHAFOLD3_TMP_ROOT"
    # wheel 构建阶段跳过 AF3 编译，后置步骤单独编译一次
    export ONESCIENCE_SKIP_AF3_BUILD=1
fi

echo ">>> Installing OneScience [$DOMAIN] with constraints"
uv pip install -c "$CONSTRAINTS_FILE" ".[${EXTRAS_SPEC}]" --system

# ==========================================
# 4. 后置环境初始化 (环境完全就绪后执行)
# ==========================================
if [[ "$DOMAIN" == "bio" || "$DOMAIN" == "all" ]]; then
    echo ">>> Building AlphaFold3 C++ extensions and data files"
    unset ONESCIENCE_SKIP_AF3_BUILD  # 取消跳过标志，允许编译
    python setup.py build_py

    # Copy built artifacts directly into site-packages (no reinstall needed)
    AF3_SRC="$SCRIPT_DIR/src/onescience/flax_models/alphafold3"
    AF3_DEST="$(python -c 'import site; print(site.getsitepackages()[0])')/onescience/flax_models/alphafold3"
    echo ">>> Copying built artifacts to $AF3_DEST"
    find "$AF3_SRC" -name "cpp*.so" -exec cp {} "$AF3_DEST/" \;
    cp -r "$AF3_SRC/_tools" "$AF3_DEST/" 2>/dev/null || true

    # cpp.so (libcifpp) looks for components.cif in site-packages/share/libcifpp/
    # at import time. Copy it there before building data files.
    if [[ -n "${ALPHAFOLD3_CIFPP_COMPONENTS:-}" && -f "$ALPHAFOLD3_CIFPP_COMPONENTS" ]]; then
        CIFPP_DEST="$(python -c 'import site; print(site.getsitepackages()[0])')/share/libcifpp"
        mkdir -p "$CIFPP_DEST"
        cp $ALPHAFOLD3_CIFPP_COMPONENTS "$CIFPP_DEST/components.cif"
        echo ">>> Copied components.cif to $CIFPP_DEST"
    fi  
fi

if [[ "$DOMAIN" == "cfd" || "$DOMAIN" == "all" ]]; then
    FASTPT_BIN="$CONDA_PREFIX/bin/fastpt"
    
    if [[ -f "$FASTPT_BIN" ]]; then
        echo ">>> Environment ready. Sourcing fastpt: $FASTPT_BIN"
        source "$FASTPT_BIN" -E || true
    else
        echo "Warning: fastpt executable not found after installation."
    fi
fi
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo ">>> Installation completed successfully!"
