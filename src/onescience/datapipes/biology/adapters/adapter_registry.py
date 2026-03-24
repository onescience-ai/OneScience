"""
适配器注册系统

管理所有模型适配器
"""

from typing import Dict, Optional, Type
import logging

from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# 适配器注册表
_adapter_registry: Dict[str, Type[BaseAdapter]] = {}


def register_adapter(name: str, adapter_cls: Type[BaseAdapter]):
    """
    注册适配器
    
    Parameters
    ----------
    name : str
        适配器名称（模型名称）
    adapter_cls : Type[BaseAdapter]
        适配器类
    """
    if not issubclass(adapter_cls, BaseAdapter):
        raise TypeError(f"{adapter_cls.__name__} must inherit from BaseAdapter")
    
    _adapter_registry[name.lower()] = adapter_cls
    logger.info(f"Registered adapter: {name} -> {adapter_cls.__name__}")


def get_adapter(name: str, config: DatasetConfig) -> BaseAdapter:
    """
    获取适配器实例
    
    Parameters
    ----------
    name : str
        适配器名称（模型名称）
    config : DatasetConfig
        数据集配置
        
    Returns
    -------
    BaseAdapter
        适配器实例
        
    Raises
    ------
    ValueError
        如果适配器不存在
    """
    name_lower = name.lower()
    
    if name_lower not in _adapter_registry:
        available = list(_adapter_registry.keys())
        raise ValueError(
            f"Adapter '{name}' not found. "
            f"Available adapters: {available}"
        )
    
    adapter_cls = _adapter_registry[name_lower]
    return adapter_cls(config)


def list_adapters() -> list[str]:
    """
    列出所有注册的适配器
    
    Returns
    -------
    list[str]
        适配器名称列表
    """
    return list(_adapter_registry.keys())

