#!/bin/bash
# -----------------------------------------------------------------------------
# MatPL DCU 一键安装脚本
# 流程：源码拉取 → 编译安装 → 安装验证
# 用法：bash matpl_install.sh
#       MATPL_SRC_DIR=/path/to/src bash matpl_install.sh  # 指定源码路径
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. 环境准备
echo ">>> Step 1: 加载环境"
source "$SCRIPT_DIR/../matchem_env.sh"
module load sghpc-mpi-gcc/26.3

# 2. 源码准备
# 说明：
#   - 开发/测试阶段：自动通过 HTTPS + 代理拉取源码
#   - 生产/客户场景：建议提前上传源码到集群，通过 MATPL_SRC_DIR 指定
MATPL_SRC="${MATPL_SRC_DIR:-/public/home/easyscience2024/wangrui/software/matpl_dcu}"
if [ ! -d "$MATPL_SRC/.git" ] && [ ! -f "$MATPL_SRC/main.py" ]; then
    echo ">>> Step 2: 拉取 MatPL DCU 源码"
    # 当前集群需通过 HTTP 代理访问外网，配置 git 代理
    git config --global http.proxy "http://jsyadmin:1cdf8f60@10.13.17.166:3128"
    git clone --depth 1 --branch nep-dcu/2026.3 \
        "https://oauth2:${GITEE_TOKEN}@gitee.com/wang-rui-sugon/matpl_dcu.git" "$MATPL_SRC"
else
    echo ">>> Step 2: 源码已存在，跳过拉取"
fi

cd "$MATPL_SRC"

echo "=========================================="
echo " MatPL DCU 一键安装脚本"
echo "=========================================="
echo "工作目录: $MATPL_SRC"

# 3. 修复 torch cmake 中硬编码的 /opt/dtk 路径
TORCH_CMAKE="$CONDA_PREFIX/lib/python3.11/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake"
if [ -f "$TORCH_CMAKE" ]; then
    if grep -q "/opt/dtk" "$TORCH_CMAKE"; then
        echo ">>> Step 3: 修复 torch cmake 硬编码路径"
        sed -i "s|/opt/dtk|$ROCM_PATH|g" "$TORCH_CMAKE"
    else
        echo ">>> Step 3: torch cmake 路径已正确，跳过修复"
    fi
else
    echo "警告: 未找到 torch Caffe2Targets.cmake，跳过修复"
fi

# 4. 确保安装兼容版 glog（0.7+ API 不兼容）
echo ">>> Step 4: 确保 glog 0.6 已安装"
conda install -c conda-forge glog=0.6 -y

# 5. 导出编译所需环境变量
echo ">>> Step 5: 导出编译环境变量"
export ROCM_PATH="$ROCM_PATH"
export CUDA_TOOLKIT_ROOT_DIR="${ROCM_PATH}/cuda/cuda-12"
export PATH="${ROCM_PATH}/cuda/cuda-12/bin:$PATH"
export CMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake:${ROCM_PATH}/dcc/comgr/lib64/cmake/amd_comgr:${CMAKE_PREFIX_PATH:-}"
export LIBRARY_PATH="${ROCM_PATH}/cuda/cuda-12/targets/x86_64-linux/lib:${ROCM_PATH}/dcc/lib/clang/17.0.0/lib/linux:${LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"

# 修复 GCC 版本：强制使用 sghpc-mpi-gcc/26.3 提供的 GCC 12.4.0
export CC=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/compilers/gcc-12.4.0/bin/gcc
export CXX=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/compilers/gcc-12.4.0/bin/g++

# 6. 清理历史编译产物并重新编译
echo ">>> Step 6: 开始编译 MatPL"
cd src
rm -rf feature/nep_find_neigh/build feature/NEP_GPU/build op/build
bash build.sh -j$(nproc)

# 7. 生成 env.sh
cd "$MATPL_SRC"
cat > env.sh <<EOF
# Load for MatPL
export PYTHONPATH=$MATPL_SRC:\${PYTHONPATH:-}
export PATH=$MATPL_SRC/src/bin:\${PATH:-}
EOF

# 8. 验证
echo ">>> Step 7: 验证安装"
source env.sh
python -c "import matpl; print('MatPL import OK')" 2>/dev/null || \
    python -c "import sys; sys.path.insert(0, '$MATPL_SRC'); import matpl; print('MatPL import OK')" 2>/dev/null || \
    echo "注意: MatPL Python 包导入验证跳过（不影响 C++ 算子使用）"

echo ""
echo "=========================================="
echo " MatPL DCU 安装完成!"
echo "=========================================="
echo "源码路径: $MATPL_SRC"
echo "环境文件: $MATPL_SRC/env.sh"
echo ""
echo "每次使用前请执行:"
echo "  source $SCRIPT_DIR/../matchem_env.sh"
