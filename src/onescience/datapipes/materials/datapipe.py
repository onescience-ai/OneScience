# onescience/datapipes/materials/datapipe.py
# (L4 "Super Factory")

import os
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from typing import Union, Dict, Any, List

# -----------------------------------------------------------------
# ✨ 关键导入 ✨
# -----------------------------------------------------------------
# 1. 导入 L1 框架核心 (Config, Datapipe基类)
from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.datapipe import Datapipe

# 2. 导入 L3-Storage (所有的 "积木")
from .storage.lmdb_dataset import LMDBDataset
from .storage.hdf5_dataset import HDF5Dataset
from .storage.text_dataset import TextDataset


class MaterialsDatapipe(Datapipe):
    """
    材料化学通用数据管道 (L4)。
    
    - 继承自 L1 (Datapipe)。
    - 负责解析 L1 (Config)，判断所需的数据格式。
    - 实例化一个或多个 L3 (Dataset) 积木。
    - 将它们拼接 (ConcatDataset) 起来。
    - 包装成 DataLoader。
    """
    
    def __init__(self, params: Union[DatasetConfig, Dict[str, Any]], distributed: bool = False, **kwargs):
        
        if isinstance(params, dict):
            params = DatasetConfig.from_dict(params) 

        meta = getattr(params, 'meta', None)
        super().__init__(meta=meta)
        
        self.params = params
        self.distributed = distributed
        self.kwargs = kwargs # 传递 heads, transform 等
        
        self.logger.info(f"MaterialsDatapipe (L4) initialized. Distributed={self.distributed}")

    def _get_dataset(self, mode: str) -> Dataset:
        """
        (核心逻辑) 移植 MACE train.py 中的“数据组装”逻辑。
        """
        
        # 1. 从 L1 Config 获取路径
        # (MACE 支持多路径，L1 Config source.path 应该是一个 List)
        file_paths = self.params.source.path
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        dataset_list = []
        for path in file_paths:
            # 2. 智能地判断文件类型
            if path.endswith(".lmdb"):
                self.logger.info(f"Loading L3-Storage: LMDBDataset (for {path})")
                # (我们需要一个轻微的技巧来传递单个路径的配置)
                db_config = self.params.to_dict()
                db_config["source"]["path"] = path
                dataset = LMDBDataset(config=db_config, mode=mode, **self.kwargs)
                dataset_list.append(dataset)
                
            elif path.endswith(".h5") or path.endswith(".hdf5"):
                self.logger.info(f"Loading L3-Storage: HDF5Dataset (for {path})")
                db_config = self.params.to_dict()
                db_config["source"]["path"] = path
                dataset = HDF5Dataset(config=db_config, mode=mode, **self.kwargs)
                dataset_list.append(dataset)
                
            elif path.endswith((".xyz", ".extxyz", ".cif")):
                self.logger.info(f"Loading L3-Storage: TextDataset (for {path})")
                db_config = self.params.to_dict()
                db_config["source"]["path"] = path
                dataset = TextDataset(config=db_config, mode=mode, **self.kwargs)
                dataset_list.append(dataset)
            
            else:
                self.logger.warning(f"Skipping unknown file type: {path}")

        # 3. (关键) 动态拼接 (Concat)
        if not dataset_list:
            raise FileNotFoundError(f"No valid datasets found in paths: {self.params.source.path}")
        
        if len(dataset_list) > 1:
            self.logger.info(f"Concatenating {len(dataset_list)} datasets into one.")
            return ConcatDataset(dataset_list)
        
        return dataset_list[0]

    def _get_dataloader(self, mode: str):
        """
        (模仿 ERA5) 创建 L3 实例和 L4 DataLoader
        """
        
        # 1. 实例化 L3 数据集 (现在是“超级组装”版本)
        dataset = self._get_dataset(mode=mode)
        
        # 2. 实例化 L4 Sampler
        sampler = None
        loader_params = getattr(self.params, 'dataloader', None)
        if loader_params is None:
            raise ValueError("DatasetConfig 中缺少 'dataloader' 配置部分")

        shuffle = loader_params.shuffle
        if shuffle is None:
            shuffle = (mode == "train") 
        
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        # 3. 实例化 L4 DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=loader_params.batch_size,
            num_workers=loader_params.num_workers,
            pin_memory=loader_params.pin_memory,
            drop_last=loader_params.drop_last or (mode != "test"),
            shuffle=shuffle,
            sampler=sampler,
            # collate_fn=... # <-- 警告：您可能需要一个自定义的 collate_fn
        )
        
        self.logger.info(f"Created '{mode}' dataloader. Batch size={loader_params.batch_size}, Num workers={loader_params.num_workers}")
            
        return data_loader, sampler

    def train_dataloader(self):
        return self._get_dataloader(mode="train")

    def val_dataloader(self):
        return self._get_dataloader(mode="val")

    def test_dataloader(self):
        return self._get_dataloader(mode="test")