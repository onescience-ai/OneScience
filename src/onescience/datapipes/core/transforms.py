"""
数据转换模块

提供可组合的数据转换流水线
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch


class Transform(ABC):
    """
    数据转换基类
    
    所有转换都应该继承此类
    """
    
    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用转换
        
        Parameters
        ----------
        data : Dict[str, Any]
            输入数据字典
            
        Returns
        -------
        Dict[str, Any]
            转换后的数据字典
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """
    组合多个转换
    
    Parameters
    ----------
    transforms : List[Transform]
        转换列表
    """
    
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """依次应用所有转换"""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class TransformPipeline:
    """
    转换流水线管理器
    
    Parameters
    ----------
    transforms : List[Transform]
        转换列表
    """
    
    def __init__(self, transforms: Optional[List[Transform]] = None):
        self.transforms = transforms or []
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用所有转换"""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def add(self, transform: Transform):
        """添加转换"""
        self.transforms.append(transform)
    
    def __len__(self) -> int:
        return len(self.transforms)
    
    def __repr__(self) -> str:
        return f"TransformPipeline(num_transforms={len(self.transforms)})"
    
    @classmethod
    def from_config(cls, transform_configs: List[Dict[str, Any]]) -> "TransformPipeline":
        """
        从配置创建转换流水线
        
        Parameters
        ----------
        transform_configs : List[Dict[str, Any]]
            转换配置列表
            
        Returns
        -------
        TransformPipeline
            转换流水线
        """
        from .transform_registry import get_transform
        
        transforms = []
        for config in transform_configs:
            transform_type = config.type if hasattr(config, 'type') else config.get('type')
            transform_params = config.params if hasattr(config, 'params') else config.get('params', {})
            
            transform_cls = get_transform(transform_type)
            if transform_cls is None:
                raise ValueError(f"Unknown transform type: {transform_type}")
            
            transforms.append(transform_cls(**transform_params))
        
        return cls(transforms)


# ============================================================================
# 常用转换
# ============================================================================

class ToTensor(Transform):
    """转换为PyTorch Tensor"""
    
    def __init__(self, keys: Optional[List[str]] = None):
        """
        Parameters
        ----------
        keys : Optional[List[str]]
            需要转换的键列表，如果为None则转换所有numpy数组
        """
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """转换numpy数组为tensor"""
        result = {}
        for key, value in data.items():
            if self.keys is None or key in self.keys:
                if isinstance(value, np.ndarray):
                    result[key] = torch.from_numpy(value)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        return f"ToTensor(keys={self.keys})"


class Normalize(Transform):
    """
    归一化数据
    
    Parameters
    ----------
    mean : Union[float, List[float], np.ndarray]
        均值
    std : Union[float, List[float], np.ndarray]
        标准差
    keys : Optional[List[str]]
        需要归一化的键
    """
    
    def __init__(
        self,
        mean: Union[float, List[float], np.ndarray],
        std: Union[float, List[float], np.ndarray],
        keys: Optional[List[str]] = None
    ):
        self.mean = np.array(mean) if not isinstance(mean, np.ndarray) else mean
        self.std = np.array(std) if not isinstance(std, np.ndarray) else std
        self.keys = keys or ["input", "target"]
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用归一化"""
        result = data.copy()
        for key in self.keys:
            if key in result:
                value = result[key]
                if isinstance(value, torch.Tensor):
                    mean = torch.tensor(self.mean, dtype=value.dtype, device=value.device)
                    std = torch.tensor(self.std, dtype=value.dtype, device=value.device)
                    result[key] = (value - mean) / std
                else:
                    result[key] = (value - self.mean) / self.std
        return result
    
    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean.tolist()}, std={self.std.tolist()}, keys={self.keys})"


class Denormalize(Transform):
    """
    反归一化数据
    
    Parameters
    ----------
    mean : Union[float, List[float], np.ndarray]
        均值
    std : Union[float, List[float], np.ndarray]
        标准差
    keys : Optional[List[str]]
        需要反归一化的键
    """
    
    def __init__(
        self,
        mean: Union[float, List[float], np.ndarray],
        std: Union[float, List[float], np.ndarray],
        keys: Optional[List[str]] = None
    ):
        self.mean = np.array(mean) if not isinstance(mean, np.ndarray) else mean
        self.std = np.array(std) if not isinstance(std, np.ndarray) else std
        self.keys = keys or ["input", "target"]
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用反归一化"""
        result = data.copy()
        for key in self.keys:
            if key in result:
                value = result[key]
                if isinstance(value, torch.Tensor):
                    mean = torch.tensor(self.mean, dtype=value.dtype, device=value.device)
                    std = torch.tensor(self.std, dtype=value.dtype, device=value.device)
                    result[key] = value * std + mean
                else:
                    result[key] = value * self.std + self.mean
        return result


class RandomCrop(Transform):
    """
    随机裁剪
    
    Parameters
    ----------
    size : Union[int, List[int]]
        裁剪大小
    keys : Optional[List[str]]
        需要裁剪的键
    """
    
    def __init__(self, size: Union[int, List[int]], keys: Optional[List[str]] = None):
        self.size = [size, size] if isinstance(size, int) else size
        self.keys = keys or ["input", "target"]
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用随机裁剪"""
        result = data.copy()
        
        # 获取第一个数据的形状来确定裁剪位置
        first_key = self.keys[0]
        if first_key not in data:
            return result
        
        value = data[first_key]
        if isinstance(value, torch.Tensor):
            h, w = value.shape[-2:]
        else:
            h, w = value.shape[-2:]
        
        th, tw = self.size
        if h < th or w < tw:
            raise ValueError(f"Data size ({h}, {w}) is smaller than crop size ({th}, {tw})")
        
        # 随机选择裁剪位置
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        
        # 对所有指定的键应用相同的裁剪
        for key in self.keys:
            if key in result:
                value = result[key]
                if isinstance(value, torch.Tensor):
                    result[key] = value[..., i:i+th, j:j+tw]
                else:
                    result[key] = value[..., i:i+th, j:j+tw]
        
        return result
    
    def __repr__(self) -> str:
        return f"RandomCrop(size={self.size}, keys={self.keys})"


class CenterCrop(Transform):
    """
    中心裁剪
    
    Parameters
    ----------
    size : Union[int, List[int]]
        裁剪大小
    keys : Optional[List[str]]
        需要裁剪的键
    """
    
    def __init__(self, size: Union[int, List[int]], keys: Optional[List[str]] = None):
        self.size = [size, size] if isinstance(size, int) else size
        self.keys = keys or ["input", "target"]
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用中心裁剪"""
        result = data.copy()
        
        for key in self.keys:
            if key in result:
                value = result[key]
                if isinstance(value, torch.Tensor):
                    h, w = value.shape[-2:]
                else:
                    h, w = value.shape[-2:]
                
                th, tw = self.size
                i = (h - th) // 2
                j = (w - tw) // 2
                
                if isinstance(value, torch.Tensor):
                    result[key] = value[..., i:i+th, j:j+tw]
                else:
                    result[key] = value[..., i:i+th, j:j+tw]
        
        return result


class Lambda(Transform):
    """
    自定义lambda转换
    
    Parameters
    ----------
    func : Callable
        转换函数
    """
    
    def __init__(self, func: Callable):
        self.func = func
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用自定义函数"""
        return self.func(data)
    
    def __repr__(self) -> str:
        return f"Lambda(func={self.func.__name__})"

