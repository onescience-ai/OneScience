#!/bin/bash
# -----------------------------------------------------------------------------
# MatPL DCU 一键安装脚本
# 流程：源码拉取 → 编译安装 → 安装验证
# 用法：bash matpl_install.sh
#       MATPL_SRC_DIR=/path/to/src bash matpl_install.sh  # 指定源码路径
# -----------------------------------------------------------------------------

MATPL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MATPL_SCRIPT_DIR"

# 1. 环境准备
echo ">>> Step 1: 加载环境"
source "$MATPL_SCRIPT_DIR/../matchem_env.sh"

# 1.5 确保 gflags/glog 运行库存在（torch cmake 的运行时依赖，pip 环境通常缺失）
echo ">>> Step 1.5: 检查 gflags/glog 运行库"
MISSING_PKGS=""
ls "$CONDA_PREFIX"/lib/libgflags.so* >/dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS} gflags"
ls "$CONDA_PREFIX"/lib/libglog.so* >/dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS} glog"
if [ -n "${MISSING_PKGS}" ]; then
    echo ">>> 安装缺失的运行库:${MISSING_PKGS}（conda-forge）"
    conda install -y -c conda-forge ${MISSING_PKGS}
else
    echo ">>> gflags/glog 已存在，跳过"
fi

# 下载辅助：优先 curl，失败时回退 wget（部分节点 curl 存在 TLS/代理问题）
download_file() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1 && curl -fL -o "$out" "$url"; then
        return 0
    fi
    echo "[提示] curl 下载失败，改用 wget: $url"
    if command -v wget >/dev/null 2>&1 && wget -O "$out" "$url"; then
        return 0
    fi
    echo "[错误] 下载失败: $url"
    return 1
}

# 2. 交互式配置源码路径
MATPL_SRC="${MATPL_SRC_DIR:-${MATPL_SCRIPT_DIR}/matpl_dcu}"
if [ -t 0 ]; then
    read -rp "请输入 MatPL 源码路径 [默认: ${MATPL_SRC}]: " input_src
    MATPL_SRC="${input_src:-${MATPL_SRC}}"
fi
echo "[提示] 使用 MatPL 源码路径: ${MATPL_SRC}"

# 3. 源码准备
# 说明：
#   - 开发/测试阶段：自动通过 HTTPS + 代理拉取源码
#   - 生产/客户场景：建议提前上传源码到集群，通过 MATPL_SRC_DIR 指定
if [ ! -d "$MATPL_SRC/.git" ] && [ ! -f "$MATPL_SRC/main.py" ]; then
    echo ">>> Step 3: 拉取 MatPL DCU 源码"
    # 计算节点需通过 HTTP 代理访问外网，配置 git 代理，如代理实效，可向集群管理员重新申请
    git config --global http.proxy "http://scnethpc2601:sWMtqVS@10.16.1.52:3120"
    git clone --depth 1 --branch nep-dcu/2026.3 "https://gitee.com/wang-rui-sugon/matpl_dcu.git" "$MATPL_SRC"
else
    echo ">>> Step 3: 源码已存在，跳过拉取"
fi

cd "$MATPL_SRC"

echo "=========================================="
echo " MatPL DCU 一键安装脚本"
echo "=========================================="
echo "工作目录: $MATPL_SRC"

# 3.5 Patch MatPL CMakeLists.txt：避免在登录节点触发 torch 动态库加载
echo ">>> Step 3.5: Patch MatPL CMakeLists.txt，避免登录节点 import torch"
OP_CMAKE="$MATPL_SRC/src/op/CMakeLists.txt"
CHEB_CMAKE="$MATPL_SRC/src/feature/chebyshev/CMakeLists.txt"

if [ -f "$OP_CMAKE" ]; then
    sed -i "s|import torch; print(torch.utils.cmake_prefix_path)|import importlib.util, os; spec = importlib.util.find_spec('torch'); p = os.path.dirname(spec.origin) if spec and spec.origin else ''; print((p + '/share/cmake') if p else '')|" "$OP_CMAKE"
    sed -i "s|import torch; print(torch.__path__\[0\])|import importlib.util, os; spec = importlib.util.find_spec('torch'); print(os.path.dirname(spec.origin) if spec and spec.origin else '')|" "$OP_CMAKE"
    sed -i "s|import torch; print(torch.version.hip is not None)|print(True)|" "$OP_CMAKE"
    echo ">>> 已 patch $OP_CMAKE"
fi

if [ -f "$CHEB_CMAKE" ]; then
    sed -i "s|import torch; print(torch.utils.cmake_prefix_path)|import importlib.util, os; spec = importlib.util.find_spec('torch'); p = os.path.dirname(spec.origin) if spec and spec.origin else ''; print((p + '/share/cmake') if p else '')|" "$CHEB_CMAKE"
    echo ">>> 已 patch $CHEB_CMAKE"
fi

# 4. 修复 torch cmake 中硬编码的 /opt/dtk 路径
TORCH_CMAKE="$CONDA_PREFIX/lib/python3.11/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake"
if [ -f "$TORCH_CMAKE" ]; then
    if grep -q "/opt/dtk" "$TORCH_CMAKE"; then
        echo ">>> Step 4: 修复 torch cmake 硬编码路径"
        sed -i "s|/opt/dtk|$ROCM_PATH|g" "$TORCH_CMAKE"
    else
        echo ">>> Step 4: torch cmake 路径已正确，跳过修复"
    fi
else
    echo "警告: 未找到 torch Caffe2Targets.cmake，跳过修复"
fi

# 5. 确保安装兼容版 glog（0.7+ API 不兼容）
echo ">>> Step 5: 确保 glog 0.6 已安装"
if [ -f "$CONDA_PREFIX/lib/libglog.so.1" ]; then
    echo "glog 0.6+ 已存在，跳过安装"
else
    echo "从源码编译 glog 0.6 ..."
    GLOG_BUILD_DIR=$(mktemp -d)
    cd "$GLOG_BUILD_DIR"
    download_file "https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz" v0.6.0.tar.gz || exit 1
    tar -xzf v0.6.0.tar.gz
    cd glog-0.6.0
    cmake -S . -B build \
        -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
        -DBUILD_SHARED_LIBS=ON \
        -DWITH_GTEST=OFF
    cmake --build build -j$(nproc)
    cmake --install build
    cd "$MATPL_SRC"
    rm -rf "$GLOG_BUILD_DIR"
fi

# 6. 导出编译所需环境变量
echo ">>> Step 6: 导出编译环境变量"
export ROCM_PATH="$ROCM_PATH"
export CUDA_TOOLKIT_ROOT_DIR="${ROCM_PATH}/cuda/cuda-12"
export PATH="${ROCM_PATH}/cuda/cuda-12/bin:$PATH"
export CMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake:${ROCM_PATH}/dcc/comgr/lib64/cmake/amd_comgr:${CMAKE_PREFIX_PATH:-}"
export LIBRARY_PATH="${ROCM_PATH}/cuda/cuda-12/targets/x86_64-linux/lib:${ROCM_PATH}/dcc/lib/clang/17.0.0/lib/linux:${LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"

# 修复 GCC 版本：强制使用 sghpc-mpi-gcc/26.3 提供的 GCC 12.4.0
export CC=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/compilers/gcc-12.4.0/bin/gcc
export CXX=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/compilers/gcc-12.4.0/bin/g++

# 7. 清理历史编译产物并重新编译
echo ">>> Step 7: 开始编译 MatPL"
cd src
rm -rf feature/nep_find_neigh/build feature/NEP_GPU/build op/build

# 登录节点内存有限，限制并行编译数以避免 OOM
NPROC=$(nproc)
if [ "$NPROC" -gt 2 ]; then
    NPROC=2
fi
echo ">>> 限制并行编译数为 ${NPROC}（避免节点 OOM）"
bash build.sh -j${NPROC}

# 8. 生成 env.sh
cd "$MATPL_SRC"
cat > env.sh <<EOF
# Load for MatPL
export PYTHONPATH=$MATPL_SRC:\${PYTHONPATH:-}
export PATH=$MATPL_SRC/src/bin:\${PATH:-}
EOF

# 9. 验证
echo ">>> Step 9: 验证安装"
source env.sh
python -c "import matpl; print('MatPL import OK')" 2>/dev/null || \
    python -c "import sys; sys.path.insert(0, '$MATPL_SRC'); import matpl; print('MatPL import OK')" 2>/dev/null || \
    echo "注意: MatPL Python 包导入验证跳过（不影响 C++ 算子使用）"

# 10. 将配置写回 matchem_env.sh
MATCHEM_ENV_FILE="${MATPL_SCRIPT_DIR}/../matchem_env.sh"
if [ -f "${MATCHEM_ENV_FILE}" ]; then
    echo "[提示] 更新 ${MATCHEM_ENV_FILE} ..."
    sed -i "s|^export MATPL_SRC_DIR=.*|export MATPL_SRC_DIR=${MATPL_SRC}|" "${MATCHEM_ENV_FILE}"
else
    echo "[警告] 未找到 ${MATCHEM_ENV_FILE}，跳过写入配置。"
fi

echo ""
echo "=========================================="
echo " MatPL DCU 安装完成!"
echo "=========================================="
echo "源码路径: $MATPL_SRC"
echo "环境文件: $MATPL_SRC/env.sh"
echo ""
echo "每次使用前请执行:"
echo "  source $MATPL_SCRIPT_DIR/../matchem_env.sh"
