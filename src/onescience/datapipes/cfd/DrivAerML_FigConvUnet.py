import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import copy
import numpy as np
import dgl
import torch
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import DictConfig, OmegaConf

from onescience.datapipes.core import BaseDataset 
from onescience.distributed.manager import DistributedManager

class DrivAerML_FigConvUnetDataset(BaseDataset):
    """
    DrivAerML Dataset tailored for FigConvUNet.
    Reads partitioned binary graph files and samples points.
    Inherits from OneScience BaseDataset.
    """
    DOMAIN = "cfd"
    DATA_FORMATS = ["bin", "json"]

    def __init__(self, config: Union[Dict[str, Any], DictConfig], mode: str = 'train'):
        self.mode = mode
        self.dist = DistributedManager()
        super().__init__(config)
        
        # Extract configuration
        if isinstance(config, DictConfig):
            self.data_cfg = config.data
            self.source_cfg = config.source
        else:
            self.data_cfg = config['data']
            self.source_cfg = config['source']

        self._init_paths()
        self._init_stats()
        
        # Set up file list based on mode
        if self.mode == 'train':
            self.partition_dir = self.data_path / "partitions"
        elif self.mode == 'val':
            self.partition_dir = self.data_path / "validation_partitions"
        elif self.mode == 'test':
            self.partition_dir = self.data_path / "test_partitions"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        self.p_files = sorted(self.partition_dir.glob("*.bin"))
        self.num_points = self.data_cfg.get('num_points', 0)
        
        if self.dist.rank != 0:
            self.logger.setLevel(logging.WARNING)
            
        if self.dist.rank == 0:
            self.logger.info(f"[{self.mode}] Loaded {len(self.p_files)} partition files from {self.partition_dir}")

    def _init_paths(self):
        """Initialize data paths from config."""
        super()._init_paths() # Sets self.data_path from config.source.data_dir
        self.stats_file = self.data_path / self.source_cfg.get('stats_filename', "global_stats.json")

    def _init_stats(self):
        """Load normalization statistics."""
        if not self.stats_file.exists():
            raise FileNotFoundError(f"Stats file not found: {self.stats_file}")
            
        with open(self.stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.mean = {k: torch.tensor(v) for k, v in stats["mean"].items()}
        self.std = {k: torch.tensor(v) for k, v in stats["std_dev"].items()}

    def encode(self, x: torch.Tensor, name: str):
        """Normalize tensor using loaded stats."""
        # Move stats to same device as input tensor on the fly
        return (x - self.mean[name].to(x.device)) / self.std[name].to(x.device)

    def decode(self, x: torch.Tensor, name: str):
        """Denormalize tensor."""
        return x * self.std[name].to(x.device) + self.mean[name].to(x.device)

    def __len__(self) -> int:
        return len(self.p_files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(f"Invalid {index = } expected in [0, {len(self)})")

        # Load DGL graphs from binary file
        gs, _ = dgl.load_graphs(str(self.p_files[index]))

        # Concatenate nodes from all partitions in the file
        coords = torch.cat([g.ndata["coordinates"] for g in gs], dim=0)
        
        # Sample indices
        n_total = coords.shape[0]
        if self.num_points > 0:
            if n_total >= self.num_points:
                indices = np.random.choice(n_total, self.num_points, replace=False)
            else:
                # If not enough points, sample all and pad with random choices
                indices = np.concatenate(
                    (
                        np.arange(n_total),
                        np.random.choice(n_total, self.num_points - n_total, replace=True),
                    )
                )
            # Shuffle to ensure randomness if constructed via concatenation
            np.random.shuffle(indices)
        else:
            # Use all points if num_points is 0
            indices = np.arange(n_total)

        coords = coords[indices]
        pressure = torch.cat([g.ndata["pressure"] for g in gs], dim=0)[indices]
        shear_stress = torch.cat([g.ndata["shear_stress"] for g in gs], dim=0)[indices]

        # Return dict matching expected input for FigConvNet
        # Note: Depending on model requirements, normalization might happen here or in the model.
        # The original code provided normalization methods in DataModule but didn't apply them in __getitem__.
        # We preserve that behavior (normalization likely happens in model or loss function).
        
        return {
            "coordinates": coords,
            "pressure": pressure,
            "shear_stress": shear_stress,
            "design": self.p_files[index].stem.removeprefix("graph_partitions_"),
            "indices": indices # Optional: useful for debugging
        }


class DrivAerML_FigConvUnetDatapipe:
    def __init__(self, config: Dict[str, Any], distributed: bool = False):
        self.config = config
        self.distributed = distributed
        if hasattr(config, "data") and hasattr(config.data, "source"):
             # 如果 config 是全局配置 (包含 train, model, data 等)
             dataset_cfg = config.data
        else:
             # 如果传入的已经是数据配置 (单元测试等情况)
             dataset_cfg = config

        # 传递 dataset_cfg 而不是 config
        self.train_dataset = DrivAerML_FigConvUnetDataset(dataset_cfg, mode='train')
        self.val_dataset = DrivAerML_FigConvUnetDataset(dataset_cfg, mode='val')
        self.test_dataset = DrivAerML_FigConvUnetDataset(dataset_cfg, mode='test')

    def _create_dataloader(self, dataset: BaseDataset, **kwargs) -> DataLoader:
        # 1. 提取基础参数
        shuffle = kwargs.pop("shuffle", False)
        sampler = kwargs.pop("sampler", None)
        
        # 2. 提取 DataLoader 核心参数 (优先 kwargs，否则使用默认值)
        # 注意：这里不再尝试去 self.config 里找，因为由调用者(train_dataloader)负责传入
        batch_size = kwargs.pop("batch_size", 1) 
        num_workers = kwargs.pop("num_workers", 0)
        pin_memory = kwargs.pop("pin_memory", True)

        if sampler is None and self.distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None) and shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

    def train_dataloader(self, **kwargs) -> DataLoader:
        # 清理 kwargs 防止冲突
        kwargs.pop("shuffle", None) 
        cfg_train = self.config.train
        
        # 提取参数，如果有 kwargs 传入则覆盖 config
        batch_size = kwargs.pop("batch_size", cfg_train.get("batch_size", 1))
        
        # 安全获取 dataloader 下的参数
        dl_cfg = cfg_train.get("dataloader", {})
        num_workers = kwargs.pop("num_workers", dl_cfg.get("num_workers", 0))
        pin_memory = kwargs.pop("pin_memory", dl_cfg.get("pin_memory", True))

        return self._create_dataloader(
            self.train_dataset, 
            shuffle=True, 
            batch_size=batch_size, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
    def val_dataloader(self, **kwargs) -> DataLoader:
        kwargs.pop("shuffle", None)
        
        # --- 核心修改：从 self.config.eval 读取配置 ---
        cfg_eval = self.config.eval
        
        batch_size = kwargs.pop("batch_size", cfg_eval.get("batch_size", 1))
        
        dl_cfg = cfg_eval.get("dataloader", {})
        num_workers = kwargs.pop("num_workers", dl_cfg.get("num_workers", 0))
        pin_memory = kwargs.pop("pin_memory", dl_cfg.get("pin_memory", True))

        return self._create_dataloader(
            self.val_dataset, 
            shuffle=False, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )

    def test_dataloader(self, **kwargs) -> DataLoader:
        # 测试集配置通常与验证集相同，也可以单独读取
        kwargs.pop("shuffle", None)
        
        cfg_eval = self.config.eval
        
        batch_size = kwargs.pop("batch_size", cfg_eval.get("batch_size", 1))
        
        dl_cfg = cfg_eval.get("dataloader", {})
        num_workers = kwargs.pop("num_workers", dl_cfg.get("num_workers", 0))
        pin_memory = kwargs.pop("pin_memory", dl_cfg.get("pin_memory", True))

        return self._create_dataloader(
            self.test_dataset, 
            shuffle=False, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
    @staticmethod
    def set_epoch(dataloader: DataLoader, epoch: int):
        """Sets the epoch for DistributedSampler."""
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)