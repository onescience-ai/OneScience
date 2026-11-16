"""
统一的生物学数据加载器接口

参考OpenFold和Protenix实现，提供通用的数据加载功能
支持蛋白质结构预测模型的训练和推理
"""

import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from onescience.datapipes.biology.adapters import ProtenixInferAdapter, BaseAdapter
from onescience.datapipes.biology.common.json import JSONParser, JSONData
from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.datasets import ProteinDataset, MultimerDataset, GenomeDataset
from onescience.utils.protenix.distributed import DIST_WRAPPER

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


# def get_biology_dataloader(configs: Any) -> DataLoader:
#     """
#     获取生物学数据加载器（工厂函数）
#     传 configs 对象（自动提取参数）
#     Creates and returns a DataLoader for inference using the InferenceDataset.
#     Args:
#         configs: A configuration object containing the necessary parameters for the DataLoader.

#     Returns:
#         A DataLoader object configured for inference.
#     """
#     # 如果传了 configs，从中提取参数
#     if configs is not None:
#         data_path = data_path or getattr(configs, 'input_json_path', None)
#         batch_size = batch_size or getattr(configs, 'batch_size', 1)
#         num_workers = num_workers or getattr(configs, 'num_workers', 0)
#         use_msa = getattr(configs, 'use_msa', use_msa)
#     if data_path is None:
#         raise ValueError("必须提供 data_path 或包含 input_json_path 的 configs")    
#     # 创建数据集
#     dataset = BiologyDataset(
#         input_json_path=configs.input_json_path,
#         dump_dir=configs.dump_dir,
#         use_msa=configs.use_msa,
#     )
    
#     # 创建数据加载器
#     sampler = DistributedSampler(
#         dataset=dataset,
#         num_replicas=DIST_WRAPPER.world_size,
#         rank=DIST_WRAPPER.rank,
#         shuffle=False,
#     )
#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=1,
#         sampler=sampler,
#         collate_fn=lambda batch: batch,
#         num_workers=configs.num_workers,
#     )
#     return dataloader


# class WeightedSampler(Sampler):
#     """
#     加权采样器（单节点）
    
#     参考Protenix实现
#     """
    
#     def __init__(
#         self,
#         weights: Sequence[float],
#         num_samples: int,
#         replacement: bool = True,
#         seed: int = 0,
#     ):
#         """
#         Args:
#             weights: 权重列表或数组
#             num_samples: 采样数量
#             replacement: 是否放回采样
#             seed: 随机种子
#         """
#         self.weights = torch.as_tensor(weights, dtype=torch.double)
#         self.replacement = replacement
#         self.seed = seed
#         self.epoch = 0
#         self.num_samples = num_samples
        
#     def __iter__(self) -> Iterator[int]:
#         """生成采样索引迭代器"""
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
#         indices = torch.multinomial(
#             self.weights, self.num_samples, self.replacement, generator=g
#         ).tolist()
#         return iter(indices)
    
#     def __len__(self) -> int:
#         return self.num_samples
    
#     def set_epoch(self, epoch: int) -> None:
#         """设置当前epoch"""
#         self.epoch = epoch


# class DistributedWeightedSampler(DistributedSampler):
#     """
#     分布式加权采样器（多节点）
    
#     参考Protenix实现
#     """
    
#     def __init__(
#         self,
#         dataset: Dataset,
#         weights: Sequence[float],
#         num_samples: int,
#         num_replicas: Optional[int] = None,
#         rank: Optional[int] = None,
#         replacement: bool = True,
#         seed: int = 0,
#     ):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
#         self.weights = torch.as_tensor(weights, dtype=torch.double)
#         self.replacement = replacement
#         self.seed = seed
#         self.epoch = 0
#         self.num_samples = num_samples
        
#         self.num_samples_per_replica = int(math.ceil(self.num_samples / self.num_replicas))
#         self.total_size = self.num_samples_per_replica * self.num_replicas
        
#     def __iter__(self) -> Iterator[int]:
#         """生成分布式采样索引迭代器"""
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
#         indices = torch.multinomial(
#             self.weights, self.num_samples, self.replacement, generator=g
#         ).tolist()
#         indices = indices[self.rank : self.total_size : self.num_replicas]
#         return iter(indices)
    
#     def __len__(self) -> int:
#         return self.num_samples // self.num_replicas
    
#     def set_epoch(self, epoch: int) -> None:
#         """设置当前epoch"""
#         self.epoch = epoch


# class BiologyDataset(Dataset):
#     """
#     生物学数据集基类
    
#     支持从JSON输入加载数据，使用适配器进行特征提取
#     """
    
#     def __init__(
#         self,
#         data_path: str,
#         adapter: Optional[BaseAdapter] = None,
#         config: Optional[DatasetConfig] = None,
#         mode: str = "train",
#         use_msa: bool = True,
#     ):
#         """
#         Args:
#             data_path: 数据文件路径（JSON格式）
#             adapter: 数据适配器（默认使用ProtenixInferAdapter）
#             config: 数据集配置
#             mode: 模式（"train"/"eval"/"predict"）
#             use_msa: 是否使用MSA特征
#         """
#         self.data_path = data_path
#         self.mode = mode
#         self.use_msa = use_msa
        
#         # 初始化适配器
#         if adapter is None:
#             config = config or DatasetConfig({
#                 "source": {"path": data_path},
#                 "data": {"extra": {"use_msa": use_msa}}
#             })
#             adapter = ProtenixInferAdapter(config)
#         self.adapter = adapter
        
#         # 加载数据
#         self.json_parser = JSONParser()
#         self._load_data()
        
#     def _load_data(self):
#         """加载JSON数据"""
#         import json
        
#         if self.data_path.endswith('.json'):
#             with open(self.data_path, 'r') as f:
#                 self.data = json.load(f)
#         else:
#             # 如果是目录，查找所有JSON文件
#             import os
#             self.data = []
#             for root, dirs, files in os.walk(self.data_path):
#                 for file in files:
#                     if file.endswith('.json'):
#                         with open(os.path.join(root, file), 'r') as f:
#                             self.data.extend(json.load(f))
                            
#         logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")
        
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         """获取单个样本"""
#         sample = self.data[idx]
        
#         try:
#             # 使用适配器处理样本
#             if hasattr(self.adapter, 'process_json_sample'):
#                 features_dict, atom_array, token_array = self.adapter.process_json_sample(sample)
#             else:
#                 features_dict = self.adapter.process_sample(sample)
#                 atom_array = None
#                 token_array = None
                
#             # 构建返回字典
#             result = {
#                 "features": features_dict,
#                 "sample_name": sample.get("name", f"sample_{idx}"),
#                 "sample_index": idx,
#             }
            
#             if atom_array is not None:
#                 result["atom_array"] = atom_array
#             if token_array is not None:
#                 result["token_array"] = token_array
                
#             return result
            
#         except Exception as e:
#             logger.error(f"Error processing sample {idx}: {e}")
#             # 返回空样本
#             return {
#                 "features": {},
#                 "sample_name": sample.get("name", f"sample_{idx}"),
#                 "sample_index": idx,
#                 "error": str(e),
#             }
            
#     def __len__(self) -> int:
#         return len(self.data)


# class BiologyDataLoader(DataLoader):
#     """
#     生物学数据加载器
    
#     支持分布式训练、加权采样、迭代式加载
#     """
    
#     def __init__(
#         self,
#         dataset: Dataset,
#         batch_size: int = 1,
#         shuffle: bool = False,
#         sampler: Optional[Sampler] = None,
#         num_workers: int = 0,
#         collate_fn: Optional[Callable] = None,
#         pin_memory: bool = False,
#         drop_last: bool = False,
#         timeout: float = 0,
#         worker_init_fn: Optional[Callable] = None,
#         multiprocessing_context=None,
#         generator=None,
#         *,
#         prefetch_factor: int = 2,
#         persistent_workers: bool = False,
#         epoch: int = 0,
#     ):
#         super().__init__(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             sampler=sampler,
#             num_workers=num_workers,
#             collate_fn=collate_fn,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#             timeout=timeout,
#             worker_init_fn=worker_init_fn,
#             multiprocessing_context=multiprocessing_context,
#             generator=generator,
#             prefetch_factor=prefetch_factor,
#             persistent_workers=persistent_workers,
#         )
#         self.epoch = epoch
        
#     def set_epoch(self, epoch: int):
#         """设置当前epoch（用于分布式采样器）"""
#         self.epoch = epoch
#         if hasattr(self.sampler, 'set_epoch'):
#             self.sampler.set_epoch(epoch)
            
#     def __iter__(self):
#         """迭代器，自动设置epoch"""
#         if hasattr(self.sampler, 'set_epoch'):
#             self.sampler.set_epoch(self.epoch)
#         self.epoch += 1
#         return super().__iter__()


# class BiologyDistributedDataLoader(BiologyDataLoader):
#     """
#     分布式生物学数据加载器
    
#     自动处理分布式采样
#     """
    
#     def __init__(
#         self,
#         dataset: Dataset,
#         batch_size: int = 1,
#         num_workers: int = 0,
#         collate_fn: Optional[Callable] = None,
#         pin_memory: bool = False,
#         drop_last: bool = True,
#         shuffle: bool = True,
#         seed: int = 42,
#         epoch: int = 0,
#     ):
#         # 创建分布式采样器
#         sampler = DistributedSampler(
#             dataset,
#             shuffle=shuffle,
#             seed=seed,
#             drop_last=drop_last,
#         )
        
#         super().__init__(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,  # 分布式采样器处理shuffle
#             sampler=sampler,
#             num_workers=num_workers,
#             collate_fn=collate_fn,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#             epoch=epoch,
#         )


# class BiologyWeightedDataLoader(BiologyDataLoader):
#     """
#     加权生物学数据加载器
    
#     支持根据样本权重进行采样
#     """
    
#     def __init__(
#         self,
#         dataset: Dataset,
#         weights: Sequence[float],
#         num_samples: int,
#         batch_size: int = 1,
#         num_workers: int = 0,
#         collate_fn: Optional[Callable] = None,
#         pin_memory: bool = False,
#         drop_last: bool = False,
#         replacement: bool = True,
#         seed: int = 42,
#         epoch: int = 0,
#         distributed: bool = False,
#     ):
#         # 创建加权采样器
#         if distributed:
#             sampler = DistributedWeightedSampler(
#                 dataset,
#                 weights=weights,
#                 num_samples=num_samples,
#                 replacement=replacement,
#                 seed=seed,
#             )
#         else:
#             sampler = WeightedSampler(
#                 weights=weights,
#                 num_samples=num_samples,
#                 replacement=replacement,
#                 seed=seed,
#             )
            
#         super().__init__(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,  # 采样器处理采样逻辑
#             sampler=sampler,
#             num_workers=num_workers,
#             collate_fn=collate_fn,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#             epoch=epoch,
#         )


def get_protein_dataloader(configs: Any) -> DataLoader:
    """
    获取蛋白质数据加载器

    Args:
        config: 数据集配置

    Returns:
        DataLoader: 配置好的数据加载器
    """
    dataset = ProteinDataset(configs)

    sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=DIST_WRAPPER.world_size,
                rank=DIST_WRAPPER.rank,
                shuffle=False,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=configs.num_workers,
    )

    return dataloader


def get_multimer_dataloader(configs: Any) -> DataLoader:
    """
    获取多聚体数据加载器

    Args:
        config: 数据集配置

    Returns:
        DataLoader: 配置好的数据加载器
    """
    dataset = MultimerDataset(configs)

    sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=DIST_WRAPPER.world_size,
                rank=DIST_WRAPPER.rank,
                shuffle=False,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=configs.num_workers,
    )

    return dataloader


def get_genome_dataloader(configs: Any) -> DataLoader:
    """
    获取基因组数据加载器

    Args:
        config: 数据集配置

    Returns:
        DataLoader: 配置好的数据加载器
    """
    dataset = GenomeDataset(configs)

    sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=DIST_WRAPPER.world_size,
                rank=DIST_WRAPPER.rank,
                shuffle=False,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=configs.num_workers,
    )

    return dataloader


# 导出
__all__ = [
    # "WeightedSampler",
    # "DistributedWeightedSampler",
    # "BiologyDataset",
    # "BiologyDataLoader",
    # "BiologyDistributedDataLoader",
    # "BiologyWeightedDataLoader",
    # "get_biology_dataloader",  # 暂未实现
    # "get_biology_weighted_dataloader",  # 暂未实现
    "get_protein_dataloader",
    "get_multimer_dataloader",
    "get_genome_dataloader",
]
