# MeshGraphNet 3D 并行训练

本项目实现了 MeshGraphNet 的 3D 并行（TP + PP + DP）训练，基于 Megatron-LM 分布式框架。

## 目录结构

```
Vortex_shedding_mgn_distributed/
├── README.md                       # 本文件
├── README_MEGATRON.md              # 完整使用指南
├── CHANGES.md                      # 详细修改清单
├── COMPLETION_SUMMARY.md           # 完成总结
├── train_megatron.py              # Megatron 训练主脚本
├── launch_megatron.sh             # 单机启动脚本
├── launch_megatron_multi_node.sh  # 多机启动脚本
└── conf/
    └── mgn_megatron.yaml          # Megatron 配置文件
```

## 核心特性

- **3D 并行支持**：数据并行（DP）+ 张量并行（TP）+ 流水线并行（PP）
- **动态 Stage 切分**：根据 PP 配置自动创建 1-3 个 Stage
- **GNN 数据结构适配**：支持 DGLGraph 的跨 Stage 通信
- **混合精度训练**：支持 BF16 混合精度
- **灵活配置**：支持 TP/PP/DP 自由组合

## 快速开始

### 1. 环境准备

确保已安装以下依赖：
- PyTorch >= 2.0
- DGL >= 0.9
- Megatron-LM (包含在 OneScience 中)

### 2. 准备数据

确保数据集路径正确，修改 `conf/mgn_megatron.yaml`:
```yaml
data-dir: ${ONESCIENCE_DATASETS_DIR}/vortex_shedding_mgn/cylinder_flow
stats-dir: ${ONESCIENCE_DATASETS_DIR}/vortex_shedding_mgn/cylinder_flow/stats
```

### 3. 单机训练

```bash
cd examples/cfd/Vortex_shedding_mgn_distributed
bash launch_megatron.sh
```

### 4. 多机训练

在主节点（node 0）上运行：
```bash
export NODE_RANK=0
export MASTER_ADDR=<主节点IP>
bash launch_megatron_multi_node.sh
```

在其他节点（node 1）上运行：
```bash
export NODE_RANK=1
export MASTER_ADDR=<主节点IP>
bash launch_megatron_multi_node.sh
```

## 并行配置示例

### 单机 8 卡

- **TP=2, PP=2, DP=2**
- 适用于中等规模模型和数据集

### 多机 16 卡

- **TP=2, PP=2, DP=4**
- 适用于大规模模型和数据集

## 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input-dim-nodes` | 输入节点特征维度 | 6 |
| `input-dim-edges` | 输入边特征维度 | 3 |
| `output-dim` | 输出特征维度 | 3 |
| `processor-size` | 消息传递层数 | 15 |
| `hidden-dim-processor` | 处理器隐藏维度 | 128 |
| `num-layers-node-processor` | 节点处理器 MLP 层数 | 2 |
| `num-layers-edge-processor` | 边处理器 MLP 层数 | 2 |
| `aggregation` | 消息聚合方式 | sum |

## 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `micro-batch-size` | 微批次大小 | 1 |
| `global-batch-size` | 全局批次大小 | 8 |
| `lr` | 学习率 | 0.0001 |
| `lr-decay-rate` | 学习率衰减率 | 0.9999991 |
| `bf16` | 使用 BF16 混合精度 | true |

## 核心代码结构

### 1. Distributed 模块

- `DistributedMeshGraphMLP`: 分布式 MLP
- `DistributedMeshEdgeBlock`: 分布式边更新块
- `DistributedMeshNodeBlock`: 分布式节点更新块

### 2. Stage 模块

- `MeshGraphNetStage0`: Encoder Stage
- `MeshGraphNetStage1`: Processor Stage
- `MeshGraphNetStage2`: Decoder Stage

### 3. 动态 Stage

- `MeshGraphNetDistributedStage`: 动态流水线 Stage
- `GNNDataStructure`: GNN 数据结构，用于 Stage 间通信

### 4. 训练脚本

- `train_megatron.py`: Megatron-LM 分布式训练主脚本

## 故障排查

### 1. 环境变量错误

确保设置了正确的环境变量：
```bash
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=6000
```

### 2. 数据路径错误

检查 `conf/mgn_megatron.yaml` 中的数据路径是否正确。

### 3. 显存不足

尝试以下方法：
- 减小 `micro-batch-size`
- 增加 `virtual-pipeline-model-parallel-size`
- 启用梯度检查点

### 4. 通信超时

增加 `NCCL_TIMEOUT` 环境变量：
```bash
export NCCL_TIMEOUT=1800
```

## 参考资料

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [MeshGraphNet 论文](https://arxiv.org/abs/2010.03409)
- [OneScience 分布式框架](https://github.com/OneScience/OneScience)