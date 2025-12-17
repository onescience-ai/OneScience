# onescience/datapipes/materials/storage/lmdb_dataset.py
# (已修复 __init__ 来适配 MACE L4 脚本的直接参数传递)

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Union, Dict, Any, List, Optional

# -----------------------------------------------------------------
# ✨ L1, L2, L3-Core, L3-Tools 导入 (保持不变) ✨
# -----------------------------------------------------------------
from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.datapipe import Datapipe
from ..base import MaterialsDataset
from ..core.atomic_data import AtomicData
from ..core.utils import KeySpecification, config_from_atoms
from ...tools.keys import DefaultKeys
from .fairchem_dataset.lmdb_dataset_tools import AseDBDataset 


# =================================================================
# 层次 3 (L3): 具体的 LMDB 数据集实现
# =================================================================

class LMDBDataset(MaterialsDataset):
    """
    LMDB 数据集实现 (L3)。
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
        #    我们将 L4 传入的参数传递给 L2
        super().__init__(
            config=None,    # L4 脚本没有提供 L1 Config，所以我们传 None
            r_max=r_max,      # <-- 将 r_max=4.0 传递给 L2
            z_table=z_table   # <-- 将 z_table 传递给 L2
        )
        
        # 2. 保留 L3 特有的初始化
        self.kwargs = kwargs
        self.kwargs["head"] = head
        self.kwargs["heads"] = heads
        self.transform = kwargs.get("transform", None)

        # 3. (self.r_max 和 self.z_table 已由 L2 基类正确设置)
        self.logger.info(f"LMDBDataset (L3) initialized with r_max={self.r_max}")

        # 4. 保留原始 MACE 的 LMDB 初始化逻辑
        #    (我们现在使用来自 __init__ 参数的 file_path)
        dataset_paths = file_path.split(":")
        for path in dataset_paths:
            if not os.path.exists(path):
                self.logger.error(f"LMDB file path does not exist: {path}")
                raise FileNotFoundError(f"LMDB file path does not exist: {path}")
        config_kwargs = {}
        self.AseDB = AseDBDataset(config=dict(src=dataset_paths, **config_kwargs))

    def __len__(self):
        """L3 必须实现 __len__"""
        return len(self.AseDB)

    def __getitem__(self, index):
        """
        L3 必须实现 __getitem__
        (此函数无需修改，因为它使用的 self.r_max 和 self.z_table 
         现在已由我们修复的 L2 基类正确设置为 4.0)
        """
        try:
            atoms = self.AseDB.get_atoms(self.AseDB.ids[index])
        except Exception as e:
            self.logger.error(f"Error reading atom index {index}: {e}")
            return None
            
        assert np.sum(atoms.get_cell() == atoms.cell) == 9

        if hasattr(atoms, "calc") and hasattr(atoms.calc, "results"):
            if "energy" in atoms.calc.results:
                atoms.info[DefaultKeys.ENERGY.value] = atoms.calc.results["energy"]
            if "forces" in atoms.calc.results:
                atoms.arrays[DefaultKeys.FORCES.value] = atoms.calc.results["forces"]
            if "stress" in atoms.calc.results:
                atoms.info[DefaultKeys.STRESS.value] = atoms.calc.results["stress"]

        config = config_from_atoms(
            atoms,
            key_specification=KeySpecification.from_defaults(),
        )

        if config.head == "Default":
            config.head = self.kwargs.get("head", "Default")

        try:
            atomic_data = AtomicData.from_config(
                config,
                z_table=self.z_table,    # <-- 来自 L2 基类 (现在是正确的)
                cutoff=self.r_max,        # <-- 来自 L2 基类 (现在是正确的 4.0)
                heads=self.kwargs.get("heads", ["Default"]),
            )
        except Exception as e:
            self.logger.error(f"Error creating AtomicData at index {index} (using {self.r_max} cutoff): {e}")
            return None

        if self.transform:
            atomic_data = self.transform(atomic_data)
        
        return atomic_data


# =================================================================
# 层次 4 (L4): Datapipe 包装器 (“休眠”状态)
# =================================================================
# (由于我们采用了方案 A (外科手术式)，这个 L4 Datapipe 类现在是“休眠”的，
#  它没有被 MACE 的 train.py 调用，您可以暂时保留它或删除它)
class MaterialsLMDBDatapipe(Datapipe):
    
    def __init__(self, params: Union[DatasetConfig, Dict[str, Any]], distributed: bool = False, **kwargs):
        if isinstance(params, dict):
            params = DatasetConfig.from_dict(params) 
        meta = getattr(params, 'meta', None) 
        super().__init__(meta=meta)
        self.params = params
        self.distributed = distributed
        self.kwargs = kwargs
        self.logger.info(f"MaterialsLMDBDatapipe (L4) initialized. Distributed={self.distributed}")

    def _get_dataloader(self, mode: str):
        dataset = LMDBDataset(config=self.params, mode=mode, **self.kwargs) # L3
        sampler = None
        shuffle = (mode == "train")
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        
        loader_params = self.params.dataloader 
        
        data_loader = DataLoader(
            dataset,
            batch_size=loader_params.batch_size,
            num_workers=loader_params.num_workers,
            pin_memory=loader_params.pin_memory,
            drop_last=(mode != "test"),
            shuffle=shuffle,
            sampler=sampler,
        )
        
        if self.distributed:
            self.logger.info(f"Created '{mode}' dataloader with DistributedSampler.")
        else:
            self.logger.info(f"Created '{mode}' dataloader. Shuffle={shuffle}")
            
        return data_loader, sampler

    def train_dataloader(self):
        return self._get_dataloader(mode="train")

    def val_dataloader(self):
        return self._get_dataloader(mode="val")

    def test_dataloader(self):
        return self._get_dataloader(mode="test")