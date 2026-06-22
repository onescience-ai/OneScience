# Fuxi

Fuxi（伏羲）是复旦大学提出的多阶段全球天气预报模型，通过 short → medium → long 三阶段级联训练，以 0.25° 分辨率实现 15 天全球预报。

> 论文：[FuXi: A cascade machine learning forecasting system for 15-day global weather forecast](https://arxiv.org/abs/2306.12873)

## 数据准备

真实数据的存储格式参照 `../era5_dataset_prepare/README.md`，在 `conf/config.yaml` 中修改：

```yaml
data_dir: 存放ERA5年度数据、均值/标准差文件、静态文件，存放方式参考'../era5_dataset_prepare/README.md'
train_time: [2000, 2001]   # 训练年份
val_time: [2002]            # 验证年份
test_time: [2003]           # 测试年份
```

无真实数据时，可生成虚拟数据快速验证流程（包含各阶段中间数据）(若快速验证，则需将conf/config.yaml中max_epoch、finetune_step同时设为1)：

```bash
source ../earth_env.sh
python fake_data.py
```

## 运行

Fuxi 包含 3 个阶段，必须按顺序执行。每个阶段的推理结果作为下一阶段的输入：

**short（推理）→ medium（推理）→ long**

```bash
source ../earth_env.sh

# 1. 训练 short 模型（从零开始，作为起始训练入口）
python train_short.py               # 单卡
# torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train_short.py   # 多卡

# 2. 推理 short（生成 medium 的输入数据）
python inference.py short

# 3. 训练 medium 模型（需要 short 权重 + short 推理结果）
python train_medium.py

# 4. 推理 medium（生成 long 的输入数据）
python inference.py medium

# 5. 训练 long 模型（需要 medium 权重 + medium 推理结果）
python train_long.py

# 6. 推理 & 评估（各阶段可独立执行）
python inference.py long
python result.py short
python result.py medium
python result.py long
```

## 集群训练

```bash
mkdir -p logs
sbatch work_slurm.sh    # 提交前检查分区、节点数等配置
```

## 许可证

Apache 2.0，可免费用于学术研究和商业用途。
