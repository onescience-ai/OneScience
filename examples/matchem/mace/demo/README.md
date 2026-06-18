# MACE 训练 Demo

## 快速上手

### 1. 选择配置

`configs/` 目录下提供了多个预定义的实验配置：

| 配置文件                        | 数据集       | GPU 数 | 说明             |
| ------------------------------- | ------------ | ------ | ---------------- |
| `DMC.yaml`                      | DMC 溶剂 XTB | 1      | 入门示例，最简单 |
| `water_1dcu.yaml`               | Water        | 1      | 单卡训练         |
| `water_4dcu.yaml`               | Water        | 4      | 4 卡分布式训练   |
| `water_8dcu.yaml`               | Water        | 8      | 8 卡分布式训练   |
| `ani1x_8dcu.yaml`               | ANI-1x       | 8      | 分布式训练       |
| `nanotube_l0_8dcu.yaml`         | 碳纳米管     | 8      | max_L=0          |
| `nanotube_l2_8dcu.yaml`         | 碳纳米管     | 8      | max_L=2          |
| `nanotube_l2_16dcu.yaml`        | 碳纳米管     | 2x8    | 多节点分布式     |

### 2. 运行训练

```bash
# 方式一：直接运行（交互式或在已分配的 SLURM 节点上）
bash run.sh --config configs/DMC.yaml

# 方式二：提交 SLURM 作业
bash run.sh --config configs/DMC.yaml --submit

# 方式三：预览命令（不执行）
bash run.sh --config configs/DMC.yaml --dry-run
```

### 3. 查看输出

训练输出自动保存到 `outputs/{实验名}_{时间戳}/` 目录，包含：
- 模型 checkpoint
- 训练日志
- 当次使用的配置快照 (`config.yaml`)

## 创建自定义实验

1. 复制一个最接近的配置文件：
   ```bash
   cp configs/DMC.yaml configs/my_experiment.yaml
   ```

2. 编辑 YAML 中的参数（只需改参数值，不用碰任何 shell 脚本）

3. 运行：
   ```bash
   bash run.sh --config configs/my_experiment.yaml
   ```

## YAML 配置字段说明

### `train_args` - 训练参数

所有字段直接映射为 `train.py` 的命令行参数。布尔值 `true` 转为标志参数（如 `swa: true` -> `--swa`），`false` 则跳过。

常用参数：

| 参数           | 说明             | 示例                          |
| -------------- | ---------------- | ----------------------------- |
| `model`        | 模型类型         | `MACE`                        |
| `r_max`        | 截断半径 (A)     | `4.0` - `6.0`                 |
| `num_channels` | 通道数           | `64`, `256`                   |
| `max_L`        | 最大角动量量子数 | `0`, `2`                      |
| `batch_size`   | 训练批大小       | `2` - `128`                   |
| `E0s`          | 原子参考能量     | `average`, `isolated`, 显式字典 |
| `swa`          | 启用随机权重平均 | `true`                        |
| `ema`          | 启用指数移动平均 | `true`                        |
| `distributed`  | 启用分布式训练   | `true` (多卡时自动添加)       |

### `launch` - 启动配置

| 参数                            | 说明       | 启动方式                       |
| ------------------------------- | ---------- | ------------------------------ |
| `num_nodes: 1, num_gpus: 1`    | 单卡       | `python train.py`              |
| `num_nodes: 1, num_gpus: N`    | 单节点多卡 | `torchrun --nproc_per_node=N`  |
| `num_nodes: M, num_gpus: N`    | 多节点     | `srun` (需 --submit)           |

### `env` - 环境配置

| 参数        | 说明                    |
| ----------- | ----------------------- |
| `conda_env` | conda 环境名            |
| `modules`   | 需要加载的 module 列表  |

### `slurm` - SLURM 作业配置

| 参数            | 说明           |
| --------------- | -------------- |
| `partition`     | SLURM 分区     |
| `time`          | 作业时间限制   |
| `cpus_per_task` | CPU 核心数     |

### `nccl` - 多节点通信配置（可选）

| 参数            | 说明                |
| --------------- | ------------------- |
| `socket_ifname` | InfiniBand 网卡名   |
| `ib_hca`        | IB HCA 设备名       |
| `proto`         | NCCL 协议           |

## 目录结构

```
demo/
  run.sh                  # 统一入口脚本
  _parse_config.py        # 配置解析器（内部使用）
  README.md               # 本文件
  configs/                # 实验配置
  templates/              # 脚本模板
    env_setup.sh          # 环境初始化
    preflight_check.sh    # 训练前预检
    slurm_header.template # SLURM header 模板
  outputs/                # 训练输出（自动创建）
```
