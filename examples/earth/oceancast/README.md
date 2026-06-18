# Oceancast

OceanCast 现在采用和旧版 OceanCast 一致的通道组织方式，目录组织尽量向 `fourcastnet` 对齐，模型直接使用 `FourCastNet`。

目录内新增了本地 [`dataloader.py`](/Users/zhao/Desktop/OneScience/onescience/examples/earth/oceancast/dataloader.py)，不再依赖外部 `data_loader_ocean`。

## 数据准备

在 `conf/config.yaml` 中修改：

```yaml
stats_dir: 均值/标准差文件路径，用于归一化
static_dir: 掩码文件路径
ocean_data_dir: 海浪数据根路径，年度 h5 文件位于 Wave_*/{year}.h5
wind_uv_dir: 原始风场根路径，年度 h5 文件位于 Wind_U10/{year}.h5、Wind_V10/{year}.h5
wind_data_dir: 预处理后风场根路径
train_time: [2000, 2001, ...]   # 训练年份
val_time: [2008]                # 验证年份
test_time: [2009]               # 测试年份
channels: [Wind_Sin, Wind_Cos, Wind_Strength, Wind_Sin_Forecast, Wind_Cos_Forecast, Wind_Strength_Forecast, Wave_Period, Wave_Height, Wave_Direction, ...]
output_channels: [Wave_Period, Wave_Height, Wave_Direction, ...]
input_types: [Wind_Sin, Wind_Cos, Wind_Strength, Wind_Sin_Forecast, Wind_Cos_Forecast, Wind_Strength_Forecast, Wave_Period, Wave_Height, Wave_Direction]
output_types: [Wave_Period, Wave_Height, Wave_Direction]
history_steps: 8
forecast_steps: 8
```

真实数据下，先生成风场派生变量：

```bash
source ../earth_env.sh
python data_preprocess.py
```

再生成均值和标准差：

```bash
python get_means_stds.py
```

无真实数据时，可生成虚拟数据快速验证流程(若快速验证，则建议将 `conf/config.yaml` 中 `max_epoch` 设为 `1`)：

```bash
source ../earth_env.sh
python fake_data.py
```

`fake_data.py` 会同时生成：

- `Wave_Period` / `Wave_Height` / `Wave_Direction` 年度 h5 文件
- `Wind_Sin` / `Wind_Cos` / `Wind_Strength` 目录
- `means_stds` 下的均值和标准差
- `artifacts/ocean_mask.npy` 掩码文件

## 运行

```bash
source ../earth_env.sh

# 1. 训练（二选一）
python train.py
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train.py

# 2. 推理（结果输出至 ./result/output/）
python inference.py

# 3. 评估 & 可视化（result.py 末尾可指定样例）
python result.py
```

## 集群训练

```bash
mkdir -p logs
sbatch work_slurm.sh
```

## 许可证

Apache 2.0，可免费用于学术研究和商业用途。
