# OneScience MatChem 环境安装与使用指南

> 本文档描述如何在 DCU 平台上从零开始搭建支持 **MACE 训练**、**DP/NEP 训练**、以及 **LAMMPS 多模型推理** 的完整 MatChem 环境。

---

## 1. 环境架构概览

| 能力 | 所属目录 | 说明 |
|------|---------|------|
| MACE 训练 | `mace/` | 已包含在 `onescience[matchem]` 基础环境中 |
| UMA 训练 | `uma/` | 已包含在 `onescience[matchem]` 基础环境中 |
| DP 训练 | `dp/` | 需额外编译安装 deepmd-kit（PyTorch 后端） |
| NEP 训练 | `nep/` | 需额外编译安装 MatPL（DCU 原生算子） |
| LAMMPS 推理 | `tools/lmp/` | 需自行编译 LAMMPS with HIP，支持 DP/NEP/MACE 后端 |

**核心原则**：所有环境变量与模块加载统一走 **`matchem_env.sh`**，子目录下的 `*_env.sh` 逐步收敛到该文件。

---

## 2. 从 Scratch 安装（分阶段）

### Step 1: 基础环境（必需）

创建 conda 环境并安装 OneScience[matchem]：

```bash
cd /public/home/easyscience2024/wangrui/onescience/examples/matchem
bash matchem_install.sh
```

该脚本会完成以下工作：
- 创建 `matchem_opt` conda 环境（Python 3.11）
- 安装 DAS 定制 PyTorch、TensorFlow、DGL 等轮子
- 安装 `onescience[matchem]`（含 MACE、UMA 及全部 Python 依赖）

完成后，每次使用环境只需：
```bash
source matchem_env.sh
```

---

### Step 2: DP 训练环境（可选）

DeepMD-kit 需从源码编译安装 PyTorch + TensorFlow 双后端：

```bash
cd dp
# 1. 获取 deepmd-kit 源码（生产环境建议提前上传，通过 DEEPMD_SRC_DIR 指定）
#    或让 dp_install.sh 自动拉取
git clone https://github.com/deepmodeling/deepmd-kit.git

# 2. 进入源码目录执行安装
bash dp_install.sh
```

**说明**：
- `dp_install.sh` 会自动检测 PyTorch 路径并启用 ROCm 后端编译
- 默认安装 **PyTorch + TensorFlow 双后端**（`DP_ENABLE_TENSORFLOW=1`），`dp --pt` 和 `dp --tf` 均可用
- C++ 接口（含 LAMMPS 插件）默认跳过编译，由开发者预编译后随 LAMMPS 安装包一起提供；如需自行编译，设置 `COMPILE_DP_CPP=1`
- 若 TensorFlow 版本 >= 2.18，编译 C++ 接口时会自动 patch `gelu_multi_device.cc`，避免内置 `Gelu` op 与 deepmd-kit 自定义 op 冲突
- 安装完成后，`dp` 命令即可使用

---

### Step 3: NEP 训练环境（可选）

NEP 基于 MatPL 框架，需编译 DCU 原生算子：

```bash
cd nep
# 1. 获取 MatPL DCU 源码（如 matpl_dcu），放置到本目录或指定路径
export MATPL_SRC_DIR=/path/to/matpl_dcu  # 默认使用 $(pwd)/matpl_dcu

# 2. 执行安装
bash matpl_install.sh
```

**说明**：
- `matpl_install.sh` 会自动生成 `dcu_install.sh` 并完成 C++ 扩展编译
- 编译依赖 `glog=0.6`、`pwdata`、`scikit-learn-intelex`，脚本会自动处理
- 安装完成后，各 NEP 算例的 `submit.sh` 中已 inline 维护 MatPL 运行时环境，无需额外加载 `matpl_env.sh`

---

### Step 4: LAMMPS 安装（推理必需，若需分子动力学推理）

OneScience 平台提供基于 DCU 适配优化后的 LAMMPS 安装包，基础版本已支持 **DP、NEP、MACE** 模型以及常用经典势函数的分子动力学推理。

**安装方式**：将安装包解压到目标目录即可，无需自行编译源码：

```bash
# 示例：解压到 ~/software/lammps_dcu
export LAMMPS_INSTALL_DIR="/path/to/your/lammps_dcu"
```

然后在 `matchem_env.sh` 中定义：

```bash
export LAMMPS_INSTALL_DIR="/path/to/your/lammps_dcu"
```

**说明**：
- 运行时**不需要暴露 LAMMPS 源码路径**，所有推理脚本通过 `matchem_env.sh` 中的 `LAMMPS_INSTALL_DIR` 获取安装目录
- `tools/lmp/lmp_script/` 下的 `cmake_hip.sh` 和 `env.sh` 仅用于**编译时**环境配置，运行时不应由 submit.sh 加载
- 运行时只需在 submit.sh 中 `source matchem_env.sh`，并显式追加必要的 `LD_LIBRARY_PATH` 即可
- 基础版本 LAMMPS 已包含 Kokkos/HIP、DP、NEP、MACE 等核心包，满足绝大多数推理需求
- DeepMD C++ 接口（`dpplugin.so`）预编译后放在独立目录（默认 `~/software/dp_cpp_dcu`），通过 `DP_CPP_DIR` 变量管理；LAMMPS 运行时需设置 `LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib`
- 若需支持其他算法或自定义势函数，请联系开发人员获取源码，并参考 `tools/lmp/lmp_script/` 下的脚本自行编译

---

## 3. 统一环境变量入口

所有子模块的运行环境应收敛到 `matchem_env.sh`：

```bash
source /public/home/easyscience2024/wangrui/onescience/examples/matchem/matchem_env.sh
```

该文件会完成：
- 加载 `sghpcdas/25.6`、`sghpc-mpi-gcc/26.3` 模块
- 激活 `matchem_opt` conda 环境
- 加载 `env.sh` 中的数据集/模型路径
- 定义 `LAMMPS_INSTALL_DIR`、`DEEPMD_SRC_DIR`、`DP_CPP_DIR`、`MATPL_SRC_DIR` 等关键路径变量

**子模块 `*_env.sh` 的定位**：
- `dp/dp_env.sh`：DeepMD 运行时环境（加载 MPI、设置后端路径）
- `tools/lmp/lmp_script/env.sh`：LAMMPS **编译时**环境（配置 DTK、torch、lib_override 等），**运行时不应由 submit.sh 加载**

DP/MACE 训练算例的 `submit.sh` 只需加载 `matchem_env.sh` 即可；NEP 算例在此基础上需在 `submit.sh` 中额外 inline 设置 MatPL 运行时环境（避免侵入公共 `matchem_env.sh`）。

---

## 4. 模型训练入口

### MACE 训练

```bash
cd mace/demo
# 直接运行
bash run.sh --config configs/DMC.yaml

# 提交 SLURM 作业
bash run.sh --config configs/DMC.yaml --submit

# 预览命令（不执行）
bash run.sh --config configs/DMC.yaml --dry-run
```

### UMA 训练

```bash
cd uma/demo
bash run.sh --config configs/oc20_ef_4dcu.yaml
```

### DP 训练

```bash
cd dp/demo/water_se_e2_a_pt
# 单卡
dp --pt train input_torch.json

# 多卡
torchrun --nproc_per_node=4 -m deepmd --pt train input_torch.json
```

### NEP 训练

```bash
cd nep/demo/nep_Cu
sbatch submit.sh   # 单卡 DCU 训练

cd nep/demo/nep_AuAg
sbatch submit.sh

cd nep/demo/nep_HfO2
sbatch submit.sh

cd nep/demo/nep_LiSiC
sbatch submit.sh
```

---

## 5. LAMMPS 推理示例

所有推理算例统一在 `tools/lmp/` 下，submit.sh 均通过 `source matchem_env.sh` 加载环境。

### DP 推理

```bash
cd tools/lmp/deepmd/dp_alloy_npt
sbatch submit.sh   # 8 卡 MPI，AlNiCuZr 合金 NPT 模拟

cd tools/lmp/deepmd/dp_pfd
sbatch submit.sh   # 8 卡 MPI，PFD 模型 NPT 模拟
```

**关键点**：
- `submit.sh` 中设置 `LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib`，LAMMPS 自动加载 `dpplugin.so`
- TensorFlow 环境变量 `TF_XLA_FLAGS` 和 `TF_ROCM_USE_IMMEDIATE_MODE` 用于绕过 DCU JIT 问题（dp_pfd 算例需要）

### NEP 推理

```bash
cd tools/lmp/nep/Cu
sbatch submit.sh   # 8 卡 MPI，Cu 体系 NVE 模拟

cd tools/lmp/nep/H2O
sbatch submit.sh   # 8 卡 MPI，H2O 体系 NPT 模拟
```

### MACE 推理

```bash
cd tools/lmp/mace/LiGaClF
sbatch submit.sh   # 8 卡 MPI，LiGaClF 体系 NPT 模拟
```

---

## 6. 目录速查

```
examples/matchem/
├── matchem_env.sh          # 统一环境入口
├── matchem_install.sh      # 基础环境安装（Step 1）
├── matchem_from_scratch.md # 搭建过程记录（开发者参考）
├── README.md               # 面向用户的完整安装指南
├── dp/
│   ├── dp_env.sh           # DP 运行环境
│   ├── dp_install.sh       # DeepMD-kit 源码安装（Step 2，含可选 C++ 编译）
│   └── demo/               # DP 训练示例
├── mace/
│   ├── demo/               # MACE 配置驱动训练入口（run.sh）
│   ├── fasteq/             # FastEq-hip 扩展安装（后续支持）
│   ├── scripts/            # 旧版命令行参考（DEPRECATED）
│   └── train.py            # MACE 训练脚本
├── nep/
│   ├── matpl_install.sh    # MatPL DCU 编译安装（Step 3）
│   └── demo/               # NEP 训练示例
│       ├── nep_AuAg/
│       ├── nep_Cu/
│       ├── nep_HfO2/
│       └── nep_LiSiC/
├── uma/
│   ├── demo/               # UMA 配置驱动训练入口
│   └── train.py            # UMA 训练脚本
├── matris/                 # MatRIS 模块测试
├── tools/
│   └── lmp/                # LAMMPS 编译与运行参考（Step 4）
│       ├── deepmd/         # DP 推理算例（dp_alloy_npt, dp_pfd）
│       ├── mace/           # MACE 推理算例
│       ├── nep/            # NEP 推理算例
│       └── lmp_script/     # LAMMPS 编译脚本（cmake_hip.sh, env.sh）
└── vasp/                   # VASP DCU 运行示例
```

**外部目录（不在 OneScience 仓库内）**：
- `~/software/deepmd-kit_dcu`：DeepMD-kit DCU 源码（`DEEPMD_SRC_DIR`）
- `~/software/lammps_dcu`：LAMMPS 安装目录（`LAMMPS_INSTALL_DIR`）
- `~/software/dp_cpp_dcu`：DeepMD C++ 接口与 LAMMPS 插件（`DP_CPP_DIR`）

---

## 6. 已知问题与注意事项

1. **DeepMD TensorFlow 后端**：DCU 平台 TensorFlow 存在 MLIR kernel_gen JIT 编译缺陷（HSACO 生成失败），PyTorch 后端训练为推荐路径；TensorFlow 后端可用于 LAMMPS 推理（已通过 `Gelu` op patch 修复 TF 2.18+ 的 `ALREADY_EXISTS` 冲突）。
2. **LAMMPS 安装路径**：`cmake_hip.sh` 中的 `CMAKE_INSTALL_PREFIX` 需根据实际安装目录修改；安装完成后在 `matchem_env.sh` 中定义 `LAMMPS_INSTALL_DIR`，运行脚本通过该变量获取路径，无需硬编码。
3. **DeepMD C++ 接口路径**：C++ 接口预编译后放在独立目录 `dp_cpp_dcu`，通过 `DP_CPP_DIR` 管理；LAMMPS+DP 推理时需设置 `LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib`。
4. **FastEq 源码依赖**：FastEq-hip 为外部私有/合作源码，需单独获取，不在 OneScience 仓库内。
5. **MatPL 源码依赖**：MatPL DCU 版本需单独获取源码，安装脚本会检查 `MATPL_SRC_DIR` 是否存在。

---

## 7. 扩展功能预览（FastEq-hip）

FastEq-hip 是 MACE 在 DCU 平台上的等变网络加速扩展，目前支持范围如下：

| 场景 | 支持状态 |
|------|---------|
| MACE 在 ASE 中的推理加速 | 已支持 |
| MACE 训练加速 | 暂不支持（后续添加） |
| MACE 在 LAMMPS 中的推理加速 | 暂不支持（后续添加） |

如需试用 ASE 推理加速，可参考 `mace/fasteq/fasteq_install.sh` 进行安装（需外部 FastEq-hip 源码）。

---

## 8. 后续优化计划（TODO）

- [x] 清理 `nep/matpl_env.sh`、`demo/run.sh`、`demo/submit.sh`，NEP 环境由各算例 `submit.sh` 自行 inline 维护
- [x] 将 `matchem_install.sh` 扩展为可选安装 DP/NEP/FastEq 的统一入口
- [x] 完善 `tools/lmp/` 的编译指南，明确 `make install` + `LAMMPS_INSTALL_DIR` 管理模式
- [x] 补充 LAMMPS+DP 推理示例与 `DP_CPP_DIR` 环境变量说明
- [ ] 补充 MatRIS 的训练/推理入口说明
