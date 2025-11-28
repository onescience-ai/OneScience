## 所有的常用数据集和用户自己构建的数据集从这里管理

from onescience.datapipes.core import BaseDataset
from onescience.datapipes.climate.era5_hdf5 import ERA5Dataset, ERA5HDF5Datapipe
from onescience.datapipes.cfd.AirfRANS import AirfRANSDataset,AirfRANSDatapipe
from onescience.datapipes.cfd.ShapeNetCar import ShapeNetCarDataset,ShapeNetCarDatapipe
from onescience.datapipes.cfd.deepmind_cylinderflow import DeepMind_CylinderFlowDataset,DeepMind_CylinderFlowDatapipe
from onescience.datapipes.cfd.eagle import EagleDataset,EagleDatapipe
from onescience.datapipes.cfd.cfdbench import CFDBenchDataset,CFDBenchDatapipe
__all__ = [
    "ERA5Dataset",
    "ERA5HDF5Datapipe",
    "AirfRANSDataset",
    "AirfRANSDatapipe",
    "ShapeNetCarDataset",
    "ShapeNetCarDatapipe",
    "DeepMind_CylinderFlowDataset",
    "DeepMind_CylinderFlowDatapipe",
    "EagleDataset",
    "EagleDatapipe",
    "CFDBenchDataset",
    "CFDBenchDatapipe",
    "BaseDataset",
]
