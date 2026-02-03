import pickle
import random
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import copy
from torch.utils.data import DataLoader, DistributedSampler
import logging
from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager

class DeepCFDDataset(BaseDataset):
    """
    DeepCFD 数据集
    加载 .pkl 文件并在内存中进行划分。
    """
    
    DOMAIN = "cfd"
    DATA_FORMATS = ["pkl"]

    def __init__(self, config: Union[Dict[str, Any]], mode: str = 'train'):
        self.mode = mode
        # 确保 DistributedManager 初始化，用于 rank 判断
        self.dist = DistributedManager()
        super().__init__(config)
        
        self._init_paths()
        self._init_data()
        
        # 控制非主进程的日志级别
        if self.dist.rank != 0:
            self.logger.setLevel(logging.WARNING)

    def _init_paths(self):
        super()._init_paths() 
        
        self.x_path = self.data_path / self.config.source.data_x_name
        self.y_path = self.data_path / self.config.source.data_y_name
        
        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(f"Data files not found at {self.data_path}")

    def _init_data(self):
        """加载 Pickle 数据，打乱顺序并根据模式划分"""
        if self.dist.rank == 0:
            self.logger.info(f"Loading raw data from {self.data_path}...")
            
        with open(self.x_path, "rb") as f:
            raw_x = pickle.load(f)
        with open(self.y_path, "rb") as f:
            raw_y = pickle.load(f)
            
        total_samples = len(raw_x)
        
        indices = list(range(total_samples))
        seed = self.config.data.seed
        random.Random(seed).shuffle(indices)
        
        self.full_y = torch.FloatTensor(raw_y) 
        
        batch = self.full_y.shape[0]
        nx = self.full_y.shape[2]
        ny = self.full_y.shape[3]
        
        self.channels_weights = (
            torch.sqrt(
                torch.mean(self.full_y.permute(0, 2, 3, 1).reshape((batch * nx * ny, 3)) ** 2, dim=0)
            )
            .view(1, -1, 1, 1)
        )
        
        # 划分数据集
        split_ratio = self.config.data.split_ratio
        split_idx = int(total_samples * split_ratio)
        
        if self.mode == 'train':
            selected_indices = indices[:split_idx]
        elif self.mode == 'test' or self.mode == 'val':
            selected_indices = indices[split_idx:]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        self.x = torch.FloatTensor(raw_x[selected_indices])
        self.y = self.full_y[selected_indices]
        
        if self.dist.rank == 0:
            self.logger.info(f"[{self.mode}] Loaded {len(self.x)} samples.")

    def get_channel_weights(self):
        """返回用于计算损失的通道权重"""
        return self.channels_weights

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.x[idx],
            "y": self.y[idx]
        }


class DeepCFDDatapipe:
    def __init__(self, config: Dict[str, Any], distributed: bool = False):
        self.config = config
        self.distributed = distributed
        
        self.train_dataset = DeepCFDDataset(copy.deepcopy(config), mode='train')
        self.test_dataset = DeepCFDDataset(copy.deepcopy(config), mode='test')

    def get_loss_weights(self):
        """获取训练所需的损失权重"""
        return self.train_dataset.get_channel_weights()

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True
        ), sampler

    def test_dataloader(self):
        # 验证/测试集通常不需要打乱
        sampler = DistributedSampler(self.test_dataset, shuffle=False) if self.distributed else None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True
        ), sampler