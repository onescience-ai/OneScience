# onescience/datapipes/materials/storage/hdf5_dataset.py
# (已修复 __init__ 来适配 MACE L4 脚本的直接参数传递)

from typing import Union, Dict, Any, List, Optional
from glob import glob
import h5py
import numpy as np

import torch
from torch.utils.data import ConcatDataset, Dataset

# -----------------------------------------------------------------
# ✨ L1, L2, L3-Core, L3-Tools 导入 (保持不变) ✨
# -----------------------------------------------------------------
from onescience.datapipes.core.config import DatasetConfig
from ..base import MaterialsDataset
from ..core.atomic_data import AtomicData
from ..core.utils import Configuration
from ...tools.utils import AtomicNumberTable


# =================================================================
# 层次 3 (L3): 具体的 HDF5 数据集实现
# =================================================================

class HDF5Dataset(MaterialsDataset):
    """
    HDF5 数据集实现 (L3)。
    """
    
    # -----------------------------------------------------------------
    # ✨ 关键修复：__init__ 签名现在与 MACE L4 脚本的调用相匹配
    #    (MACE L4 脚本: run_train_utils.py/load_dataset_for_path)
    # -----------------------------------------------------------------
    def __init__(self, 
                 file_path: str,  # <-- MACE L4 传递的参数
                 r_max: float,      # <-- MACE L4 传递的参数
                 z_table: Any,      # <-- MACE L4 传递的参数
                 head: str = "Default", # <-- MACE L4 传递的参数
                 heads: Optional[List[str]] = None,
                 **kwargs):

        # 1. ✨ 关键：调用 L2 基类 (base.py)
        super().__init__(
            config=None,    # L4 脚本没有提供 L1 Config，所以我们传 None
            r_max=r_max,      # <-- 将 r_max=4.0 传递给 L2
            z_table=z_table   # <-- 将 z_table 传递给 L2
        )
        
        # 2. 保留 L3 特有的初始化
        self.kwargs = kwargs
        self.kwargs["head"] = head
        self.kwargs["heads"] = heads
        
        self.file_path = file_path
        self._file = None # 延迟加载 HDF5 文件句柄

        # 3. (self.r_max 和 self.z_table 已由 L2 基类正确设置)
        self.logger.info(f"HDF5Dataset (L3) initialized with r_max={self.r_max}")
            
        # 4. 保留原始 MACE 的 HDF5 初始化逻辑
        batch_key = list(self.file.keys())[0] # self.file 是一个 @property
        self.batch_size = len(self.file[batch_key].keys())
        self.length = len(self.file.keys()) * self.batch_size
        
        try:
            self.drop_last = bool(self.file.attrs["drop_last"])
        except KeyError:
            self.drop_last = False

    @property
    def file(self):
        """延迟打开 HDF5 文件句柄，以支持多进程 (num_workers > 0)"""
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        """允许 Dataloader (num_workers > 0) pickle 数据集对象"""
        _d = dict(self.__dict__)
        _d["_file"] = None # 文件句柄不能被 pickle，在 worker 进程中重新打开
        return _d

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        L3 必须实现 __getitem__
        (此函数无需修改，因为它使用的 self.r_max 和 self.z_table 
         现在已由我们修复的 L2 基类正确设置为 4.0)
        """
        
        # (索引逻辑保持不变)
        batch_index = index // self.batch_size
        config_index = index % self.batch_size
        
        try:
            grp = self.file["config_batch_" + str(batch_index)]
            subgrp = grp["config_" + str(config_index)]
        except Exception as e:
            self.logger.error(f"HDF5 read error at index {index}: {e}")
            return None

        # (属性加载逻辑保持不变)
        properties = {}
        property_weights = {}
        for key in subgrp["properties"]:
            properties[key] = unpack_value(subgrp["properties"][key][()])
        for key in subgrp["property_weights"]:
            property_weights[key] = unpack_value(subgrp["property_weights"][key][()])

        # (调用 L3-Core Configuration 保持不变)
        config = Configuration(
            atomic_numbers=subgrp["atomic_numbers"][()],
            positions=subgrp["positions"][()],
            properties=properties,
            weight=unpack_value(subgrp["weight"][()]),
            property_weights=property_weights,
            config_type=unpack_value(subgrp["config_type"][()]),
            pbc=unpack_value(subgrp["pbc"][()]),
            cell=unpack_value(subgrp["cell"][()]),
        )
        
        if config.head is None:
            config.head = self.kwargs.get("head")

        try:
            # (调用 L3-Core AtomicData 保持不变)
            # 它现在将使用正确的 self.r_max 和 self.z_table
            atomic_data = AtomicData.from_config(
                config,
                z_table=self.z_table,    # <-- 来自 L2 基类 (现在是正确的)
                cutoff=self.r_max,        # <-- 来自 L2 基类 (现在是正确的 4.0)
                heads=self.kwargs.get("heads", ["Default"]),
                **{k: v for k, v in self.kwargs.items() if k != "heads"},
            )
        except Exception as e:
            self.logger.error(f"Error creating AtomicData at index {index} (using {self.r_max} cutoff): {e}")
            return None
            
        return atomic_data

# -----------------------------------------------------------------
# ✨ 关键修复：修复 sharded_hdf5 辅助函数
# -----------------------------------------------------------------
def dataset_from_sharded_hdf5(
    files: str, # MACE L4 传入的是目录路径 (str)
    r_max: float,      # <-- MACE L4 传递的参数
    z_table: Any,      # <-- MACE L4 传递的参数
    **kwargs
):
    """
    (这是 MACE 原始代码中的辅助函数，
     我们必须修复它以使用新的 L3 __init__ 签名)
    """
    file_list = glob(files + "/*")
    datasets = []
    
    if not file_list:
        logging.warning(f"No HDF5 files found in sharded directory: {files}")
        
    for file in file_list:
        # ✨ 关键：我们现在将 r_max 和 z_table 传递给 L3 构造函数
        datasets.append(HDF5Dataset(
            file_path=file, 
            r_max=r_max, 
            z_table=z_table, 
            **kwargs
        ))
        
    if not datasets:
        return [] # 返回一个空列表，而不是崩溃
        
    full_dataset = ConcatDataset(datasets)
    return full_dataset


def unpack_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value