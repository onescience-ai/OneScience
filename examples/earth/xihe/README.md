# Xihe

Xihe（羲和）是面向高分辨率全球海洋预报的 Transformer 模型，输入包含海表高度、海表温度、10m 风场以及多层海温、盐度和流速变量。

## 数据准备

真实数据的存储格式参照 `../era5_dataset_prepare/README.md`，在 `conf/config.yaml` 中修改：

```yaml
data_dir: ./data/                  # 年度文件位于 data_dir/data/{year}.h5
stats_dir: ./data/stats/           # global_means.npy / global_stds.npy
mask: ./data/land_mask.npy         # 海陆掩码
train_time: [2010, 2011, 2012]
val_time: [2013]
test_time: [2014]
```

无真实数据时，可生成虚拟数据快速验证流程(若快速验证，则需将conf/config.yaml中max_epoch设为1)：

```bash
source ../earth_env.sh
python fake_data.py
```

## 运行

```bash
source ../earth_env.sh

# 1. 训练
python train.py
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train.py

# 2. 推理
python inference.py

# 3. 评估与可视化
python result.py
```

## 集群训练

```bash
mkdir -p logs
sbatch work_slurm.sh
```

## 许可证

Apache 2.0。
