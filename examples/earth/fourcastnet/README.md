# FourCastNet

FourCastNet（Fourier ForeCasting Neural Network）是基于 AFNO 的全球天气预报模型，以 0.25° 分辨率提供中短期预报，推理速度比传统 NWP 快数个数量级。

> 论文：[FourCastNet: A Global Data-driven High-resolution Weather Forecasting Model](https://arxiv.org/abs/2202.11214)

## 数据准备

真实数据的存储格式参照 `../era5_dataset_prepare/README.md`，在 `conf/config.yaml` 中修改：

```yaml
data_dir: 存放ERA5年度数据、均值/标准差文件、静态文件，存放方式参考'../era5_dataset_prepare/README.md'
train_time: [2000, 2001]   # 训练年份
val_time: [2002]            # 验证年份
test_time: [2003]           # 测试年份
```

无真实数据时，可生成虚拟数据快速验证流程(若快速验证，则需将conf/config.yaml中max_epoch设为1)：

```bash
source ../earth_env.sh
python fake_data.py
```

## 运行

```bash
source ../earth_env.sh

# 1. 训练（二选一）
python train.py                # 单卡
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train.py   # 多卡

# 2. 推理（结果输出至 ./result/output/）
python inference.py

# 3. 评估 & 可视化（result.py 末尾可指定日期和变量）
python result.py
```

## 集群训练

```bash
mkdir -p logs
sbatch work_slurm.sh    # 提交前检查分区、节点数等配置
```

## 许可证

Apache 2.0，可免费用于学术研究和商业用途。
