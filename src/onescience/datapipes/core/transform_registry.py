"""
Transform注册机制

允许用户注册和发现数据转换
"""

from typing import Dict, Optional, Type
from .transforms import Transform


# 全局转换注册表
_TRANSFORM_REGISTRY: Dict[str, Type[Transform]] = {}


def register_transform(name: str):
    """
    注册转换装饰器
    
    Parameters
    ----------
    name : str
        转换名称
        
    Examples
    --------
    >>> @register_transform("MyTransform")
    >>> class MyTransform(Transform):
    >>>     def __call__(self, data):
    >>>         return data
    """
    def decorator(cls: Type[Transform]):
        if not issubclass(cls, Transform):
            raise TypeError(f"{cls.__name__} must inherit from Transform")
        _TRANSFORM_REGISTRY[name] = cls
        return cls
    return decorator


def get_transform(name: str) -> Optional[Type[Transform]]:
    """
    获取注册的转换类
    
    Parameters
    ----------
    name : str
        转换名称
        
    Returns
    -------
    Optional[Type[Transform]]
        转换类，如果不存在则返回None
    """
    return _TRANSFORM_REGISTRY.get(name)


def list_transforms() -> Dict[str, Type[Transform]]:
    """
    列出所有注册的转换
    
    Returns
    -------
    Dict[str, Type[Transform]]
        转换字典
    """
    return _TRANSFORM_REGISTRY.copy()


# 注册内置转换
from .transforms import (
    ToTensor, Normalize, Denormalize,
    RandomCrop, CenterCrop, Lambda, Compose
)

_TRANSFORM_REGISTRY.update({
    "ToTensor": ToTensor,
    "Normalize": Normalize,
    "Denormalize": Denormalize,
    "RandomCrop": RandomCrop,
    "CenterCrop": CenterCrop,
    "Lambda": Lambda,
    "Compose": Compose,
})

