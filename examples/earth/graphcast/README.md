# GraphCast

GraphCast 是 Google DeepMind 提出的基于图神经网络（GNN）的全球中期天气预报模型，以编码-处理-解码架构实现高效的全球信息传递，在 0.25° 分辨率上支持多步自回归预报。

> 论文：[GraphCast: Learning skillful medium-range global weather forecasting](https://arxiv.org/abs/2212.12794)

## 数据准备

真实数据的存储格式参照 `../era5_dataset_prepare/README.md`，在 `conf/config.yaml` 中修改：

```yaml
data_dir: 存放ERA5年度数据、均值/标准差文件、静态文件，存放方式参考'../era5_dataset_prepare/README.md'
train_time: [2000, 2001]   # 训练年份
val_time: [2002]            # 验证年份
test_time: [2003]           # 测试年份
```

无真实数据时，可生成虚拟数据快速验证流程(若快速验证，则需将conf/config.yaml中max_epoch、num_iters_step3设为1)：

```bash
source ../earth_env.sh
python fake_data.py
```

数据准备好后，还需生成 GraphCast 专用的辅助文件：

```bash
python compute_time_diff_std.py
python get_data_json.py
```

## 运行

```bash
source ../earth_env.sh

# 1. 训练（二选一）
python train.py                # 单卡
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train.py   # 多卡

# 2. 微调（二选一）
python finetune.py             # 单卡
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 finetune.py   # 多卡

# 3. 推理（结果输出至 ./result/output/）
python inference.py

# 4. 评估 & 可视化（result.py 末尾可指定日期和变量）
python result.py
```

## 集群训练

```bash
mkdir -p logs
sbatch work_slurm.sh    # 提交前检查分区、节点数等配置
```

## 许可证

Apache 2.0，可免费用于学术研究和商业用途。
