# Pangu-3D MoE 改造说明
基于 Pangu-3D 实现 MoE（混合专家）模型并行训练，**当前仅支持 PP + EP 并行**，TP + EP 混合并行将在后续版本支持。

---


## 1. 环境配置
启动前需设置 CUDA 环境变量：
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

---

## 2. 一键训练脚本
训练入口：直接运行 `train_moe.sh` 启动 PP+EP 并行训练。

---

## 3. 启动方式
```bash
chmod +x train_moe.sh
./train_moe.sh
```

---

## 4. 关键参数说明

### MoE 相关
| 参数 | 说明 |
|------|------|
| `--num-experts` | 专家总数 |
| `--moe-router-topk` | 每个 token 路由到 K 个专家 |
| `--moe-ffn-hidden-size` | MoE 专家 FFN 隐层维度 |
| `--moe-router-load-balancing-type` | 负载均衡策略，支持 `sinkhorn` 等 |

### 并行配置
| 参数 | 说明 |
|------|------|
| `--tensor-model-parallel-size` | 张量并行大小，当前固定为 1 |
| `--pipeline-model-parallel-size` | 流水线并行（PP）大小 |
| `--expert-model-parallel-size` | 专家并行（EP）大小 |
| `--expert-tensor-parallel-size` | 专家内 TP，当前固定为 1 |
| `--moe-token-dispatcher-type` | token 调度方式，使用 `alltoall` |


---

## 5. 并行约束
- 总 GPU 数 = PP × EP
  示例：`PP=2`、`EP=4` → 总卡数 8
- 专家数必须能被 `expert-model-parallel-size` 整除
- 现阶段 TP 必须设为 1，TP+EP 组合暂不支持

