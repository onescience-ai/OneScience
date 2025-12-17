# ERA5 Data Downloader

该项目提供在通过气候数据存储 (CDS) API 下载 ERA5 数据集、通过海洋数据存储(CMEMS)API下载海洋生化再分西数据集并将其处理为适合机器学习格式的工具

## 文件描述

1. `start_mirror.py` -首先初始化“ERA5Mirror”类，该类将负责下载 ERA5 数据并将其保存为 Zarr 格式以及进行 HDF5 转换。
2. `era5_mirror.py` - 包含负责从 CDS API 下载 ERA5 数据集并将其存储为 Zarr 格式的 ERA5Mirror 类。
3. `conf/config_tas.yaml` - `start_mirror.py` 的配置文件，规定了下载和处理的参数。此配置文件将仅下载表面温度变量，但是，如果您想要更完整的数据集，例如用于训练 [FourCastNet](https://arxiv.org/abs/2202.11214) 的数据集，使用 `conf/config_34var.yaml`.
4. `ocean_mirror.py` -负责下载nc格式海洋生化数据，降采样海洋生化数据、以及HDF5转换。
 
## 数据下载

1. 确保已按照[这些说明](https://cds.climate.copernicus.eu/api-how-to) 设置 CDS API 密钥。
2. 运行主脚本“python start_mirror.py”以生成训练所需的 HDF5 文件。首先，它将下载并将所有变量保存为 Zarr 数组。如果该过程中断，它会保存下载过程的状态，然后可以重新启动，重新启动时应保持相同的日期配置。下载完成后，所需的变量将以标准格式保存为 HDF5 文件。
3. 按照`ocean_mirror.py`文件内的要求安装包和注册账号

## 配置文件

配置文件包含可以修改的配置，

- `zarr_store_path`: Zarr 数据集的保存路径。
- `hdf5_store_path`: HDF5 数据集的保存路径。
- `dt`: 时间分辨率。
- `start_train_year`: 训练数据的开始年份。
- `end_train_year`: 训练数据的结束年份。
- `test_years`: 测试数据的年份列表。
- `out_of_sample_years`: 样本外数据的年份列表。
- `compute_mean_std`: 是否计算全局平均值和标准差。
- `variables`: 需要下载的 ERA5 变量。

## 注意

请务必配置 CDS API 密钥。始终保密并避免将其推送到公共存储库。
