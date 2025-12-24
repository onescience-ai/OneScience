## 所有的常用数据集和用户自己构建的数据集从这里管理

from onescience.datapipes.core import BaseDataset
# from onescience.datapipes.cfd.AirfRANS import AirfRANSDataset, AirfRANSDatapipe
# from onescience.datapipes.cfd.ShapeNetCar import ShapeNetCarDataset, ShapeNetCarDatapipe
# from onescience.datapipes.cfd.deepmind_cylinderflow import DeepMind_CylinderFlowDataset, DeepMind_CylinderFlowDatapipe
# from onescience.datapipes.cfd.eagle import EagleDataset, EagleDatapipe
# from onescience.datapipes.cfd.cfdbench import CFDBenchDataset, CFDBenchDatapipe
# from onescience.datapipes.cfd import DeepCFDDataset, DeepCFDDatapipe
# from onescience.datapipes.cfd import BENODataset, BENODatapipe
# from onescience.datapipes.cfd import (
#     PDEBenchFNODataset,
#     PDEBenchFNODatapipe,
#     PDEBenchDeepONetDataset,
#     PDEBenchDeepONetDatapipe,
#     PDEBenchMPNNDataset,
#     PDEBenchMPNNDatapipe,
#     PDEBenchUNetDataset,
#     PDEBenchUNetDatapipe,
#     PDEBenchUNODataset,
#     PDEBenchUNODatapipe,
#     PDEBenchPINODataset,
#     PDEBenchPINODatapipe,
# )
# from onescience.datapipes.cfd import DeepMindLagrangianDataset, DeepMindLagrangianDatapipe
from onescience.datapipes.climate import ERA5Dataset, ERA5Datapipe, ERA5HDF5Datapipe, ERA5HDF5Dataset, CMEMSDatapipe, CMEMSHDF5Dataset

__all__ = [
    "ERA5Dataset",
    "ERA5HDF5Datapipe",
    # "AirfRANSDataset",
    # "AirfRANSDatapipe",
    # "ShapeNetCarDataset",
    # "ShapeNetCarDatapipe",
    # "DeepMind_CylinderFlowDataset",
    # "DeepMind_CylinderFlowDatapipe",
    # "EagleDataset",
    # "EagleDatapipe",
    # "CFDBenchDataset",
    # "CFDBenchDatapipe",
    # "BaseDataset",
    # "ERA5Datapipe",
    # "ERA5HDF5Dataset",
    # "DeepCFDDataset",
    # "DeepCFDDatapipe",
    # "PDEBenchFNODataset",
    # "PDEBenchFNODatapipe",
    # "PDEBenchDeepONetDataset",
    # "PDEBenchDeepONetDatapipe",
    # "PDEBenchMPNNDataset",
    # "PDEBenchMPNNDatapipe",
    # "PDEBenchUNetDataset",
    # "PDEBenchUNetDatapipe",
    # "PDEBenchUNODataset",
    # "PDEBenchUNODatapipe",
    # "PDEBenchPINODataset",
    # "PDEBenchPINODatapipe",
    # "DeepMindLagrangianDataset",
    # "DeepMindLagrangianDatapipe",
    # "BENODataset",
    # "BENODatapipe",
    "CMEMSDatapipe",
    "CMEMSHDF5Dataset",
]