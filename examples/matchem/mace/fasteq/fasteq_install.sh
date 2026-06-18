#!/bin/bash
# FastEq-hip 扩展安装脚本（OneScience MACE matchem 环境）
# 前置条件：已安装好 OneScience 的 matchem 环境。
# 本脚本安装 FastEq-hip 及 patch 后的 cuequivariance-torch，
# 以实现 MACE 模型在 DCU 上的训练和推理加速。
#
# 使用方法：
#   cd /public/home/easyscience2024/wangrui/onescience/examples/matchem/mace/fasteq 
#   bash install.sh

set -e

# =============================================================================
# 1. 环境初始化
# =============================================================================
# 必须先加载 module，再激活 conda；否则 sghpcdas 会覆盖 conda 环境变量。
source ~/.bashrc

module load sghpcdas/25.6
module load sghpc-mpi-gcc/26.3

conda activate matchem_opt

# libgalaxyhip.so.5 位于 DTK 的 lib/ 和 hip/lib/ 下，但不在默认 LD_LIBRARY_PATH 中。
# 不加此路径会导致导入 torch/fasteq 时报错：
#   ImportError: libgalaxyhip.so.5: cannot open shared object file
export LD_LIBRARY_PATH="/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib:/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/hip/lib:${LD_LIBRARY_PATH}"

# DTK 的 cmake 配置查找 amd_comgr，但 CMAKE_PREFIX_PATH 默认为空，需要显式指定。
export CMAKE_PREFIX_PATH="/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/dcc/comgr/lib64/cmake/amd_comgr:${CMAKE_PREFIX_PATH}"

# FastEq-hip 编译需要显式声明后端为 HIP，否则默认回退到 CUDA 导致找不到 nvcc。
export FASTEQ_BACKEND=hip
export ROCM_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4
export HIP_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4

# CMake 3.26 的 HIP 编译器 ABI 检测阶段（try_compile）不会继承主项目的 cache 变量，
# 但会读取 <Package>_DIR 环境变量。若不设置，会因找不到 amd_comgr 而报错。
export amd_comgr_DIR=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/dcc/comgr/lib64/cmake/amd_comgr

# =============================================================================
# 2. FastEq-hip 源码路径
# =============================================================================
# 默认路径：与 onescience 项目同级目录。
# 可通过环境变量覆盖：FASTEQ_HIP_ROOT=/your/path bash install.sh
FASTEQ_HIP_ROOT="${FASTEQ_HIP_ROOT}"

if [[ ! -d "$FASTEQ_HIP_ROOT" ]]; then
    echo "错误：FastEq-hip 源码未找到于 $FASTEQ_HIP_ROOT"
    echo "请先克隆仓库，或通过 FASTEQ_HIP_ROOT 环境变量指定正确路径。"
    exit 1
fi

echo ">>> FastEq-hip 源码路径: $FASTEQ_HIP_ROOT"

# =============================================================================
# 3. 安装 FastEq-hip
# =============================================================================
# PYTHONNOUSERSITE=1 避免使用 ~/.local 下的旧版 setuptools（59.8.0），
# 该版本会导致 editable_wheel 构建失败（KeyError: 'unix_user'）。
#
# 注意：FastEq-hip 的 setup.py 中 HIP 后端的 extra_cmake_args 需补充两项：
#   -DCMAKE_HIP_COMPILER=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/llvm/bin/clang++
#   -Damd_comgr_DIR=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/dcc/comgr/lib64/cmake/amd_comgr
# 原因：CMake 3.26 不支持将 hipcc wrapper 直接作为 CMAKE_HIP_COMPILER，
# 必须使用 ROCm 的 clang++；且 try_compile 阶段不会继承 cache 变量中的 amd_comgr_DIR，
# 需要通过环境变量（已在上方导出）或 cmake 参数显式传递。
echo ">>> 正在安装 FastEq-hip（editable 模式）..."
PYTHONNOUSERSITE=1 pip install -r "$FASTEQ_HIP_ROOT/requirements.txt"
PYTHONNOUSERSITE=1 pip install -e "$FASTEQ_HIP_ROOT" --no-build-isolation

# =============================================================================
# 4. 安装 patch 后的 cuequivariance-torch
# =============================================================================
# cuequivariance==0.8.0 基础包已由 OneScience matchem 安装。
# 此处只需覆盖安装本地子模块中的 HIP patch 版本。
echo ">>> 正在安装 patched cuequivariance-torch（editable 模式）..."
PYTHONNOUSERSITE=1 pip install -e "$FASTEQ_HIP_ROOT/3rdparty/cuequivariance_torch/cuequivariance_torch" --no-build-isolation

# =============================================================================
# 5. 验证安装
# =============================================================================
echo ">>> 验证导入..."
python -c "import fasteq; print('  fasteq:', fasteq.__file__)"
python -c "import cuequivariance_torch as cuet; print('  cuequivariance_torch:', cuet.__file__)"

echo ">>> FastEq-hip 扩展安装成功！"
echo ">>> 现在可以运行 MACE 训练，并传入 --enable_cueq 启用 FastEq 加速。"
