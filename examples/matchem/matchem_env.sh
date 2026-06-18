#!/bin/bash
# ==========================================
# MatChem 统一环境配置脚本
# 用途：加载模块、激活 conda、导出各组件路径
# 用法：source matchem_env.sh
# ==========================================

# ---------- 1. 基础环境配置 ----------
export MATCHEM_CONDA_NAME=matchem_opt          # conda 环境名
export ONESCIENCE_MAIN_DIR=/public/home/easyscience2024/wangrui/onescience  # OneScience 源码根目录

# ---------- 2. 训练软件源码/安装路径 ----------
export DEEPMD_SRC_DIR=/public/home/easyscience2024/wangrui/software/deepmd-kit_dcu   # DeepMD-kit 源码
export MATPL_SRC_DIR="${MATPL_SRC_DIR:-/public/home/easyscience2024/wangrui/software/matpl_dcu}"  # MatPL 源码

# ---------- 3. LAMMPS 与 C++ 接口路径 ----------
export LAMMPS_SRC_DIR="${LAMMPS_SRC_DIR:-/public/home/easyscience2024/wangrui/MD/lammps/lammps_0426}"        # LAMMPS 源码目录（编译 C++ 接口时需要）
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-/public/home/easyscience2024/wangrui/software/lammps_dcu}"  # LAMMPS 安装目录
export DP_CPP_DIR="${DP_CPP_DIR:-/public/home/easyscience2024/wangrui/software/dp_cpp_dcu_v3}"                  # DeepMD C++ 接口安装目录(pytorch和tensoflow双框架)

# ---------- 4. 加载集群模块与 conda ----------
source ~/.bashrc
module load sghpcdas/25.6        # DTK / PyTorch 等 SDK
module load sghpc-mpi-gcc/26.3   # MPI 与 GCC 编译器

conda activate $MATCHEM_CONDA_NAME

# ---------- 5. 加载 OneScience 环境变量 ----------
source $ONESCIENCE_MAIN_DIR/env.sh
