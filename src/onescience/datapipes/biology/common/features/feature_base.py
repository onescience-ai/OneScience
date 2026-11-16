"""
特征处理基类

定义通用的特征处理接口和基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Any, TypeVar, Generic
import numpy as np
import torch

# 类型别名
FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


class BaseFeatureExtractor(ABC):
    """
    特征提取器基类
    
    所有特征提取器都应该继承这个类
    """
    
    @abstractmethod
    def extract(self, data: Any) -> FeatureDict:
        """
        提取特征
        
        Parameters
        ----------
        data : Any
            输入数据
            
        Returns
        -------
        FeatureDict
            特征字典
        """
        pass
    
    def __call__(self, data: Any) -> FeatureDict:
        """方便调用"""
        return self.extract(data)


class FeaturePipeline(ABC):
    """
    特征处理管道基类
    
    定义特征处理流程的通用接口
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Parameters
        ----------
        config : Dict[str, Any]
            配置字典
        """
        self.config = config or {}
    
    @abstractmethod
    def process(self, raw_features: FeatureDict) -> FeatureDict:
        """
        处理原始特征
        
        Parameters
        ----------
        raw_features : FeatureDict
            原始特征字典
            
        Returns
        -------
        FeatureDict
            处理后的特征字典
        """
        pass
    
    def __call__(self, raw_features: FeatureDict) -> FeatureDict:
        """方便调用"""
        return self.process(raw_features)


class ModelAdapter(ABC):
    """
    模型适配器基类
    
    将通用特征转换为特定模型需要的格式
    """
    
    @abstractmethod
    def adapt(self, common_features: FeatureDict) -> FeatureDict:
        """
        将通用特征转换为模型特定格式
        
        Parameters
        ----------
        common_features : FeatureDict
            通用特征字典
            
        Returns
        -------
        FeatureDict
            模型特定格式的特征字典
        """
        pass
    
    def __call__(self, common_features: FeatureDict) -> FeatureDict:
        """方便调用"""
        return self.adapt(common_features)


class FeatureMerger:
    """
    特征合并器
    
    用于合并多个特征字典
    """
    
    @staticmethod
    def merge(
        feature_dicts: list,
        allow_overlap: bool = False,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        合并多个特征字典
        
        Parameters
        ----------
        feature_dicts : list
            特征字典列表
        allow_overlap : bool
            是否允许键重叠，如果为False且有重叠则报错
        strict : bool
            如果为True，要求所有字典的数组形状兼容
            
        Returns
        -------
        Dict[str, Any]
            合并后的特征字典
        """
        merged = {}
        
        for features in feature_dicts:
            if features is None:
                continue
                
            for key, value in features.items():
                if key in merged:
                    if not allow_overlap:
                        if strict:
                            raise ValueError(f"Duplicate key found: {key}")
                        else:
                            continue
                merged[key] = value
                
        return merged
    
    @staticmethod
    def merge_with_priority(
        feature_dicts: list,
        priority: list = None
    ) -> Dict[str, Any]:
        """
        按优先级合并特征字典
        
        Parameters
        ----------
        feature_dicts : list
            特征字典列表
        priority : list
            优先级列表，数值越大优先级越高
            
        Returns
        -------
        Dict[str, Any]
            合并后的特征字典
        """
        if priority is None:
            priority = list(range(len(feature_dicts)))
            
        # 按优先级排序
        sorted_pairs = sorted(
            zip(feature_dicts, priority),
            key=lambda x: x[1],
            reverse=True
        )
        
        merged = {}
        for features, _ in sorted_pairs:
            if features is None:
                continue
            # 后面的（优先级高的）会覆盖前面的
            merged.update(features)
            
        return merged


class FeatureFilter:
    """
    特征过滤器
    
    用于选择和过滤特征
    """
    
    @staticmethod
    def select(
        features: Dict[str, Any],
        keys: list
    ) -> Dict[str, Any]:
        """
        选择指定键的特征
        
        Parameters
        ----------
        features : Dict[str, Any]
            特征字典
        keys : list
            要选择的键列表
            
        Returns
        -------
        Dict[str, Any]
            选择后的特征字典
        """
        return {k: features[k] for k in keys if k in features}
    
    @staticmethod
    def exclude(
        features: Dict[str, Any],
        keys: list
    ) -> Dict[str, Any]:
        """
        排除指定键的特征
        
        Parameters
        ----------
        features : Dict[str, Any]
            特征字典
        keys : list
            要排除的键列表
            
        Returns
        -------
        Dict[str, Any]
            过滤后的特征字典
        """
        return {k: v for k, v in features.items() if k not in keys}
    
    @staticmethod
    def filter_by_shape(
        features: Dict[str, np.ndarray],
        required_shape: tuple
    ) -> Dict[str, np.ndarray]:
        """
        根据形状过滤特征
        
        Parameters
        ----------
        features : Dict[str, np.ndarray]
            特征字典
        required_shape : tuple
            要求的形状（支持-1表示任意维度）
            
        Returns
        -------
        Dict[str, np.ndarray]
            过滤后的特征字典
        """
        filtered = {}
        for key, value in features.items():
            if not isinstance(value, np.ndarray):
                continue
            if len(value.shape) != len(required_shape):
                continue
            match = True
            for i, (s, r) in enumerate(zip(value.shape, required_shape)):
                if r != -1 and s != r:
                    match = False
                    break
            if match:
                filtered[key] = value
        return filtered
