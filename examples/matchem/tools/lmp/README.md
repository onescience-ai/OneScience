# LAMMPS DCU 预编译安装包说明

## 预编译包下载地址

| 组件 | 下载地址 | 说明 |
|------|----------|------|
| LAMMPS + DP/NEP/MACE | `https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/lammps_dcu.tar.gz` | 含 HIP 后端、ML-MACE、NEP_KK、ML-DEEPMD 插件 |
| DeepMD C++ 接口 | `https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/dp_cpp_dcu.tar.gz` | 含 `dpplugin.so` 及 ROCm/TF 算子库 |

## 安装路径

解压后放置到以下目录（与 `matchem_env.sh` 中的变量保持一致）：

```bash
# LAMMPS 安装目录
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-~/software/lammps_dcu}"

# DeepMD C++ 接口目录
export DP_CPP_DIR="${DP_CPP_DIR:-~/software/dp_cpp_dcu}"
```

## 快速安装

```bash
# 1. LAMMPS
mkdir -p ~/software/lammps_dcu
cd ~/software
curl -L -o lammps_dcu.tar.gz \
  "https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/lammps_dcu.tar.gz"
tar -xzf lammps_dcu.tar.gz -C ~/software/lammps_dcu --strip-components=1
rm -f lammps_dcu.tar.gz

# 2. DeepMD C++ 接口
mkdir -p ~/software/dp_cpp_dcu
cd ~/software
curl -L -o dp_cpp_dcu.tar.gz \
  "https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/dp_cpp_dcu.tar.gz"
tar -xzf dp_cpp_dcu.tar.gz -C ~/software/dp_cpp_dcu --strip-components=1
rm -f dp_cpp_dcu.tar.gz
```

## 运行时环境

加载 `matchem_env.sh` 后，通过 `LD_LIBRARY_PATH` 和 `LAMMPS_PLUGIN_PATH` 加载依赖：

```bash
source /path/to/matchem_env.sh
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib_override:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib
```

## 验证安装

```bash
ldd ${LAMMPS_INSTALL_DIR}/bin/lmp_mpi | grep "not found"
# 期望无输出，表示所有依赖库均可解析

${LAMMPS_INSTALL_DIR}/bin/lmp_mpi -h | head -n 5
# 应正常输出 LAMMPS 帮助信息
```

## 自行编译（可选）

如需从源码编译 LAMMPS，参考 `script/cmake_hip.sh`：

```bash
cd <lammps_source_dir>
mkdir -p build_unified && cd build_unified
bash ../script/cmake_hip.sh
make -j$(nproc)
make install
```

关键 CMake 选项：
- `-DPKG_ML-MACE=on`
- `-DPKG_ML-DEEPMD=on`
- `-DPKG_NEP_KK=yes`
- `-DGPU_API=hip`
