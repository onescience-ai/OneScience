#!/bin/bash
# -----------------------------------------------------------------------------
# DeepMD-kit DCU 一键安装脚本
# 流程：源码拉取 → 编译安装 → 安装验证
# 用法：bash dp_install.sh
#       DEEPMD_SRC_DIR=/path/to/src bash dp_install.sh  # 指定源码路径
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
#   - 生产/客户场景：建议提前上传源码到集群，通过 DEEPMD_SRC_DIR 指定
DEEPMD_SRC="${DEEPMD_SRC_DIR:-/public/home/easyscience2024/wangrui/software/deepmd-kit_dcu}"
if [ ! -d "$DEEPMD_SRC/.git" ] && [ ! -f "$DEEPMD_SRC/setup.py" ]; then
    echo ">>> Step 2: 拉取 DeepMD-kit 源码"
    # 当前集群需通过 HTTP 代理访问外网，配置 git 代理
    git config --global http.proxy "http://jsyadmin:1cdf8f60@10.13.17.166:3128"
    git clone --depth 1 "https://oauth2:${GITEE_TOKEN}@gitee.com/wang-rui-sugon/deepmd-kit_dcu.git" "$DEEPMD_SRC"
else
    echo ">>> Step 2: 源码已存在，跳过拉取"
fi

# 3. 预先锁定 numpy 版本，避免 deepmd-kit 安装过程中短暂升级到不兼容版本
echo ">>> Step 3: 预先锁定 numpy 版本"
pip install numpy==1.26.3 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 4. 修复 Torch cmake 硬编码 DTK 路径
echo ">>> Step 4: 修复 Torch cmake 硬编码 DTK 路径"
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
CAFFE2_CMAKE="${TORCH_PATH}/share/cmake/Caffe2/Caffe2Targets.cmake"
DTK_REAL_PATH="/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4"
if [ -f "$CAFFE2_CMAKE" ] && grep -q '/opt/dtk' "$CAFFE2_CMAKE"; then
    echo ">>> 替换 Caffe2Targets.cmake 中的 /opt/dtk 为实际路径"
    sed -i "s|/opt/dtk|${DTK_REAL_PATH}|g" "$CAFFE2_CMAKE"
fi

# 5. 编译安装（PyTorch + TensorFlow 双后端）
echo ">>> Step 5: 编译安装 Python 包（PyTorch + TensorFlow 双后端）"
cd "$DEEPMD_SRC"
DP_VARIANT=rocm \
ROCM_ROOT="$ROCM_PATH" \
DP_ENABLE_TENSORFLOW=1 \
DP_ENABLE_PYTORCH=1 \
PYTORCH_ROOT="${TORCH_PATH}" \
    pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 6. 验证
echo ">>> Step 6: 验证安装"
dp -h | head -n 5

echo "========================================"
echo " DeepMD-kit Python 包安装完成"
echo "========================================"

# 7. C++ 接口安装（含 LAMMPS 插件）
# 说明：默认从预编译包下载解压，快速部署；如需自行源码编译，设置 COMPILE_DP_CPP=1。
DP_CPP_URL="https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/dp_cpp_dcu.tar.gz"

if [ "${COMPILE_DP_CPP:-0}" = "1" ]; then
    echo ">>> Step 7: 源码编译 C++ 接口（含 LAMMPS 插件）"

    # 7.1 Patch Gelu op：TensorFlow 2.18+ 已内置 Gelu，与 deepmd-kit 自定义 op 冲突，
    #     需在编译前注释掉 source/op/tf/gelu_multi_device.cc 中的 REGISTER_OP("Gelu")
    #     和 REGISTER_OP("GeluGrad") 及其属性链。（GeluGradGrad / GeluCustom 系列不受影响）
    GELU_FILE="$DEEPMD_SRC/source/op/tf/gelu_multi_device.cc"
    if grep -q '^REGISTER_OP("Gelu")' "$GELU_FILE"; then
        echo ">>> Step 7.1: Patch Gelu op 注册，避免 TF 2.18+ 冲突"
        sed -i '/^REGISTER_OP("Gelu")$/,/^);$/{ /^$/!s/^/\/\/ /; }' "$GELU_FILE"
        sed -i '/^REGISTER_OP("GeluGrad")$/,/^);$/{ /^$/!s/^/\/\/ /; }' "$GELU_FILE"
    fi

    cd "$DEEPMD_SRC/source"
    mkdir -p build && cd build

    cmake -DENABLE_TENSORFLOW=ON \
          -DENABLE_PYTORCH=ON \
          -DUSE_ROCM_TOOLKIT=ON \
          -DTENSORFLOW_ROOT="/public/home/easyscience2024/.conda/envs/matchem_opt/lib/python3.11/site-packages/tensorflow" \
          -DTensorFlow_INCLUDE_DIRS="/public/home/easyscience2024/.conda/envs/matchem_opt/lib/python3.11/site-packages/tensorflow/include" \
          -DTorch_DIR="/public/home/easyscience2024/.conda/envs/matchem_opt/lib/python3.11/site-packages/torch/share/cmake/Torch" \
          -DHIP_ROOT_DIR="/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/hip" \
          -DCMAKE_PREFIX_PATH="/public/home/easyscience2024/.conda/envs/matchem_opt;/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib/cmake" \
          -DLAMMPS_SOURCE_ROOT="${LAMMPS_SRC_DIR}" \
          -DCMAKE_INSTALL_PREFIX="${DP_CPP_DIR}" \
          ..

    make -j$(nproc)
    make install

    # 后处理：创建 dpplugin.so 符号链接
    cd "${DP_CPP_DIR}/lib"
    if [ -f "deepmd_lmp/dpplugin.so" ] && [ ! -e "dpplugin.so" ]; then
        ln -s deepmd_lmp/dpplugin.so ./
    fi
else
    echo ">>> Step 7: 下载预编译 C++ 接口包"
    mkdir -p "${DP_CPP_DIR}"
    cd "${DP_CPP_DIR}"
    curl -L -o dp_cpp_dcu.tar.gz "${DP_CPP_URL}"
    tar -xzf dp_cpp_dcu.tar.gz --strip-components=1
    rm -f dp_cpp_dcu.tar.gz
    echo ">>> Step 7: C++ 接口安装完成（${DP_CPP_DIR}）"
fi
