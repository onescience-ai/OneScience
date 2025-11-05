"""
BaseDataset - 所有OneScience数据集的统一基类

提供标准化的数据集接口和通用功能
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import cached_property

from torch.utils.data import Dataset

from .config import DatasetConfig
from .transforms import TransformPipeline


class BaseDataset(Dataset, ABC):
    """
    OneScience数据集基类
    
    所有数据集都应该继承此类，提供统一的接口
    
    Parameters
    ----------
    config : DatasetConfig
        数据集配置对象
        
    Attributes
    ----------
    config : DatasetConfig
        数据集配置
    transform : TransformPipeline
        数据转换流水线
    logger : logging.Logger
        日志记录器
    """
    
    # 元数据 - 子类应该覆盖这些属性
    DOMAIN = "base"  # earth, cfd, biology, materials, structural
    TASK = None  # forecasting, classification, regression, etc.
    REQUIRED_PARAMS = []  # 必需的配置参数
    OPTIONAL_PARAMS = []  # 可选的配置参数
    DATA_FORMATS = []  # 支持的数据格式
    
    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
    # def __init__(self):
        super().__init__()
        
        # 配置处理
        if isinstance(config, dict):
            self.config = DatasetConfig.from_dict(config)
        else:
            self.config = config
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 验证配置
        # self._validate_config()
        
        # 初始化Transform pipeline
        self.transform = None
        if self.config.transforms:
            self.transform = TransformPipeline.from_config(self.config.transforms)
        
        # 数据集状态
        self._initialized = False
        self._num_samples = 0
        
        # 子类需要实现的初始化
        self._init_paths()
        self._load_metadata()
        self._init_data()
        
        self._initialized = True
        
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self)} samples")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"onescience.datapipes.{self.__class__.__name__}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        
        return logger
    
    def _validate_config(self):
        """验证配置参数"""
        # 检查必需参数
        for param in self.REQUIRED_PARAMS:
            if not hasattr(self.config.data, param):
                raise ValueError(f"Missing required parameter: {param}")
        
        # 检查数据格式
        if self.DATA_FORMATS and self.config.source.type not in self.DATA_FORMATS:
            raise ValueError(
                f"Unsupported data format: {self.config.source.type}. "
                f"Supported formats: {self.DATA_FORMATS}"
            )
        
        # 检查路径
        if isinstance(self.config.source.path, (str, Path)):
            path = Path(self.config.source.path)
            if not path.exists():
                self.logger.warning(f"Data path does not exist: {path}")
    
    @abstractmethod
    def _init_paths(self):
        """
        初始化数据路径
        
        子类必须实现此方法，用于设置数据文件路径等
        """
        pass
    
    # @abstractmethod
    # def _load_metadata(self):
    #     """
    #     加载元数据
        
    #     子类必须实现此方法，用于加载数据集的元数据
    #     """
    #     pass
    
    # @abstractmethod
    # def _init_data(self):
    #     """
    #     初始化数据
        
    #     子类必须实现此方法，用于初始化数据加载相关的配置
    #     """
    #     pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        子类必须实现此方法
        
        Parameters
        ----------
        idx : int
            样本索引
            
        Returns
        -------
        Dict[str, Any]
            样本数据字典，包含input, target等字段
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
    
    # @cached_property
    # def metadata(self) -> Dict[str, Any]:
    #     """
    #     数据集元数据
        
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         元数据字典
    #     """
    #     return {}
    
    # def get_sample_info(self, idx: int) -> Dict[str, Any]:
    #     """
    #     获取样本信息（不加载实际数据）
        
    #     Parameters
    #     ----------
    #     idx : int
    #         样本索引
            
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         样本信息
    #     """
    #     return {
    #         "index": idx,
    #         "dataset": self.__class__.__name__,
    #         "domain": self.DOMAIN,
    #         "task": self.TASK,
    #     }
    
    # def get_statistics(self) -> Dict[str, Any]:
    #     """
    #     获取数据集统计信息
        
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         统计信息
    #     """
    #     return {
    #         "num_samples": len(self),
    #         "domain": self.DOMAIN,
    #         "task": self.TASK,
    #         "split": self.config.source.split,
    #     }
    
    # def __repr__(self) -> str:
    #     """字符串表示"""
    #     return (
    #         f"{self.__class__.__name__}("
    #         f"domain={self.DOMAIN}, "
    #         f"task={self.TASK}, "
    #         f"split={self.config.source.split}, "
    #         f"num_samples={len(self)})"
    #     )
    
    # def describe(self) -> str:
    #     """
    #     详细描述数据集
        
    #     Returns
    #     -------
    #     str
    #         数据集详细信息
    #     """
    #     info = [
    #         f"Dataset: {self.__class__.__name__}",
    #         f"Domain: {self.DOMAIN}",
    #         f"Task: {self.TASK}",
    #         f"Split: {self.config.source.split}",
    #         f"Number of samples: {len(self)}",
    #         f"Data path: {self.config.source.path}",
    #     ]
        
    #     if self.config.data.variables:
    #         info.append(f"Variables: {', '.join(self.config.data.variables)}")
        
    #     if self.transform:
    #         info.append(f"Transforms: {len(self.transform.transforms)} transforms")
        
    #     return "\n".join(info)


# class CachedDataset(BaseDataset):
#     """
#     支持缓存的数据集基类
    
#     在第一次访问时将数据缓存到内存中，适用于小规模数据集
#     """
    
#     def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
#         self._cache: Dict[int, Any] = {}
#         self._use_cache = config.cache if isinstance(config, DatasetConfig) else config.get("cache", False)
#         super().__init__(config)
    
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         """获取样本，使用缓存"""
#         if self._use_cache and idx in self._cache:
#             return self._cache[idx]
        
#         data = self._load_sample(idx)
        
#         if self.transform is not None:
#             data = self.transform(data)
        
#         if self._use_cache:
#             self._cache[idx] = data
        
#         return data
    
#     @abstractmethod
#     def _load_sample(self, idx: int) -> Dict[str, Any]:
#         """
#         加载单个样本（不使用缓存）
        
#         子类必须实现此方法
#         """
#         pass
    
#     def clear_cache(self):
#         """清空缓存"""
#         self._cache.clear()
#         self.logger.info("Cache cleared")
    
#     def preload(self):
#         """预加载所有数据到缓存"""
#         if not self._use_cache:
#             self.logger.warning("Cache is disabled, cannot preload")
#             return
        
#         self.logger.info(f"Preloading {len(self)} samples...")
#         for idx in range(len(self)):
#             if idx not in self._cache:
#                 self._cache[idx] = self._load_sample(idx)
#         self.logger.info("Preloading completed")


# class LazyDataset(BaseDataset):
#     """
#     延迟加载数据集基类
    
#     只在需要时加载数据，适用于大规模数据集
#     """
    
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         """获取样本，延迟加载"""
#         data = self._load_sample(idx)
        
#         if self.transform is not None:
#             data = self.transform(data)
        
#         return data
    
#     @abstractmethod
#     def _load_sample(self, idx: int) -> Dict[str, Any]:
#         """
#         加载单个样本
        
#         子类必须实现此方法
#         """
#         pass

