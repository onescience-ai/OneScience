## 所有的常用数据集和用户自己构建的数据集从这里管理

from onescience.datapipes.core import BaseDataset
from onescience.datapipes.climate.era5_hdf5_new import ERA5Dataset, ERA5HDF5Datapipe

__all__ = [
    "ERA5Dataset",
    "ERA5HDF5Datapipe",
    "BaseDataset",
]
