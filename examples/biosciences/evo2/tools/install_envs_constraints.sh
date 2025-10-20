#!/bin/bash

# 定义要安装的包及其对应的文件映射关系
declare -A package_map
package_map["torch==2.4.1+das.opt1.dtk25041"]="torch-2.4.1+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl"
package_map["torchvision==0.19.1+das.opt2.dtk2504"]="torchvision-0.19.1+das.opt2.dtk2504-cp311-cp311-manylinux_2_28_x86_64.whl"
package_map["apex==1.4.0+das.opt2.dtk2504"]="apex-1.4.0+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl"
package_map["jax==0.4.34+das.opt1.dtk25041"]="jax-0.4.34+das.opt1.dtk25041-py3-none-any.whl"
package_map["jax-rocm60-pjrt==0.4.34+das.opt1.dtk25041"]="jax_rocm60_pjrt-0.4.34+das.opt1.dtk25041-py3-none-any.whl"
package_map["jax-rocm60-plugin==0.4.34+das.opt1.dtk25041"]="jax_rocm60_plugin-0.4.34+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl"
package_map["jaxlib==0.4.34+das.opt1.dtk25041"]="jaxlib-0.4.34+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl"
package_map["onnxruntime==1.15.0+das.opt1.dtk2504"]="onnxruntime-1.15.0+das.opt1.dtk2504-cp311-cp311-linux_x86_64.whl"
package_map["jax-triton==0.2.0+das.opt1.dtk25041"]="jax_triton-0.2.0+das.opt1.dtk25041-py3-none-any.whl"
package_map["triton==3.0.0+das.opt1.dtk25041"]="triton-3.0.0+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl"
# package_map["triton==3.0.0+das.opt1.dtk25041"]="triton-3.0.0+das.opt4.dtk2504-cp311-cp311-manylinux_2_28_x86_64.whl"
package_map["transformer-engine==1.9.0+das.opt2.dtk2504"]="transformer_engine-1.9.0+das.opt2.dtk2504-cp311-cp311-manylinux_2_28_x86_64.whl"
package_map["flash_attn==2.6.1+das.opt1.dtk25041"]="flash_attn-2.6.1+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl"

# 定义安装顺序（有依赖关系的包）
install_order=(
    "jax-rocm60-pjrt==0.4.34+das.opt1.dtk25041"
    "jax-rocm60-plugin==0.4.34+das.opt1.dtk25041"
)

# 安装目录，根据实际情况修改
INSTALL_DIR="/public/home/onescience2025404/packages"
# INSTALL_DIR="/work/home/onescience2025/packages"

for pkg in "${install_order[@]}"; do
    whl_file="${package_map[$pkg]}"
    whl_path="$INSTALL_DIR/$whl_file"

    if [ -f "$whl_path" ]; then
        echo ">>> [Priority Install] Installing $pkg from $whl_path"
        pip install "$whl_path"
    else
        echo "!!! Warning: File $whl_path not found for priority package $pkg."
    fi
done

# 安装剩余的包（自动跳过已安装）
for package in "${!package_map[@]}"; do
    # 跳过已在 install_order 中的包
    skip=false
    for ordered_pkg in "${install_order[@]}"; do
        if [[ "$package" == "$ordered_pkg" ]]; then
            skip=true
            break
        fi
    done
    if $skip; then
        continue
    fi

    whl_file="${package_map[$package]}"
    whl_path="$INSTALL_DIR/$whl_file"
    if [ -f "$whl_path" ]; then
        echo ">>> Installing $package from $whl_path"
        pip install "$whl_path"
    else
        echo "!!! Warning: File $whl_path for package $package not found. Skipping."
    fi
done