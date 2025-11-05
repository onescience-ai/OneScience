"""
数据集工厂

提供统一的数据集创建接口，支持多种创建方式
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from ..core.base_dataset import BaseDataset
from ..core.config import DatasetConfig, SourceConfig, DataConfig, TransformConfig
from .dataset_registry import DatasetRegistry


logger = logging.getLogger("onescience.datapipes.factory")


class DatasetFactory:
    """
    数据集工厂类
    
    提供多种方式创建数据集实例
    """
    
    @staticmethod
    def create(
        name: str,
        config: Optional[Union[DatasetConfig, Dict[str, Any], str, Path]] = None,
        **kwargs
    ) -> BaseDataset:
        """
        创建数据集实例
        
        Parameters
        ----------
        name : str
            数据集名称（在注册表中的名称）
        config : Optional[Union[DatasetConfig, Dict, str, Path]]
            配置，可以是：
            - DatasetConfig对象
            - 配置字典
            - YAML/JSON配置文件路径
            - None (使用kwargs构建配置)
        **kwargs
            额外的配置参数，会覆盖config中的值
            
        Returns
        -------
        BaseDataset
            数据集实例
            
        Examples
        --------
        >>> # 方式1：使用kwargs
        >>> dataset = DatasetFactory.create(
        ...     name="era5",
        ...     path="/data/era5",
        ...     split="train",
        ...     variables=["t2m", "u10"],
        ... )
        
        >>> # 方式2：使用配置字典
        >>> config = {
        ...     "source": {"path": "/data/era5", "split": "train"},
        ...     "data": {"variables": ["t2m", "u10"]},
        ... }
        >>> dataset = DatasetFactory.create("era5", config=config)
        
        >>> # 方式3：使用配置文件
        >>> dataset = DatasetFactory.create("era5", config="config.yaml")
        """
        # 获取数据集类
        dataset_cls = DatasetRegistry.get(name)
        if dataset_cls is None:
            available = DatasetRegistry.list()
            raise ValueError(
                f"Dataset '{name}' not found in registry. "
                f"Available datasets: {available}"
            )
        
        # 处理配置
        final_config = DatasetFactory._build_config(name, config, **kwargs)
        
        # 创建实例
        logger.info(f"Creating dataset: {name}")
        dataset = dataset_cls(final_config)
        
        return dataset
    
    @staticmethod
    def _build_config(
        name: str,
        config: Optional[Union[DatasetConfig, Dict[str, Any], str, Path]] = None,
        **kwargs
    ) -> DatasetConfig:
        """
        构建数据集配置
        
        优先级：kwargs > config > 默认值
        """
        # 获取数据集信息
        dataset_info = DatasetRegistry.get_info(name)
        
        # 基础配置
        base_config = {
            "name": name,
            "domain": dataset_info.domain if dataset_info else "unknown",
            "task": dataset_info.task if dataset_info else None,
        }
        
        # 处理config参数
        if config is None:
            config_dict = base_config
        elif isinstance(config, DatasetConfig):
            return config
        elif isinstance(config, dict):
            config_dict = {**base_config, **config}
        elif isinstance(config, (str, Path)):
            # 从文件加载
            config_path = Path(config)
            if config_path.suffix in [".yaml", ".yml"]:
                config_dict = DatasetConfig.from_yaml(config_path).to_dict()
            elif config_path.suffix == ".json":
                config_dict = DatasetConfig.from_json(config_path).to_dict()
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            config_dict = {**base_config, **config_dict}
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
        
        # 处理kwargs，构建嵌套结构
        if kwargs:
            config_dict = DatasetFactory._merge_kwargs(config_dict, **kwargs)
        
        # 创建DatasetConfig对象
        return DatasetConfig.from_dict(config_dict)
    
    @staticmethod
    def _merge_kwargs(config_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        将kwargs合并到配置字典中
        
        自动将扁平的kwargs转换为嵌套结构
        """
        # 初始化嵌套字典
        if "source" not in config_dict:
            config_dict["source"] = {}
        if "data" not in config_dict:
            config_dict["data"] = {}
        if "transforms" not in config_dict:
            config_dict["transforms"] = []
        
        # source相关参数
        source_keys = ["path", "split", "type", "format", "metadata_path"]
        # data相关参数
        data_keys = [
            "variables", "features", "channels",
            "time_range", "time_steps", "time_resolution",
            "spatial_resolution", "spatial_crop",
            "num_samples", "shuffle", "random_seed",
        ]
        # 顶层参数
        top_keys = ["name", "domain", "task", "verbose", "cache", "cache_dir"]
        
        for key, value in kwargs.items():
            if key in source_keys:
                config_dict["source"][key] = value
            elif key in data_keys:
                config_dict["data"][key] = value
            elif key in top_keys:
                config_dict[key] = value
            elif key == "transforms":
                config_dict["transforms"] = value
            else:
                # 其他参数放入data.extra
                if "extra" not in config_dict["data"]:
                    config_dict["data"]["extra"] = {}
                config_dict["data"]["extra"][key] = value
        
        return config_dict
    
    @classmethod
    def create_from_config_file(
        cls,
        config_path: Union[str, Path],
        **kwargs
    ) -> BaseDataset:
        """
        从配置文件创建数据集
        
        Parameters
        ----------
        config_path : Union[str, Path]
            配置文件路径
        **kwargs
            额外参数，会覆盖配置文件中的值
            
        Returns
        -------
        BaseDataset
            数据集实例
        """
        config_path = Path(config_path)
        
        # 加载配置
        if config_path.suffix in [".yaml", ".yml"]:
            config = DatasetConfig.from_yaml(config_path)
        elif config_path.suffix == ".json":
            config = DatasetConfig.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # 创建数据集
        return cls.create(config.name, config=config, **kwargs)
    
    @classmethod
    def create_multi(
        cls,
        configs: list[Union[str, Dict[str, Any], DatasetConfig]],
        **common_kwargs
    ) -> list[BaseDataset]:
        """
        批量创建多个数据集
        
        Parameters
        ----------
        configs : List[Union[str, Dict, DatasetConfig]]
            数据集配置列表
        **common_kwargs
            所有数据集共享的参数
            
        Returns
        -------
        List[BaseDataset]
            数据集实例列表
        """
        datasets = []
        
        for config in configs:
            if isinstance(config, str):
                # 假设是数据集名称
                dataset = cls.create(config, **common_kwargs)
            elif isinstance(config, dict):
                name = config.pop("name")
                merged_kwargs = {**common_kwargs, **config}
                dataset = cls.create(name, **merged_kwargs)
            elif isinstance(config, DatasetConfig):
                dataset = cls.create(config.name, config=config, **common_kwargs)
            else:
                raise TypeError(f"Unsupported config type: {type(config)}")
            
            datasets.append(dataset)
        
        logger.info(f"Created {len(datasets)} datasets")
        return datasets


# 便捷函数
def create_dataset(
    name: str,
    config: Optional[Union[DatasetConfig, Dict[str, Any], str, Path]] = None,
    **kwargs
) -> BaseDataset:
    """
    创建数据集（便捷函数）
    
    Parameters
    ----------
    name : str
        数据集名称
    config : Optional[Union[DatasetConfig, Dict, str, Path]]
        配置
    **kwargs
        额外配置参数
        
    Returns
    -------
    BaseDataset
        数据集实例
        
    Examples
    --------
    >>> from onescience.datapipes import create_dataset
    
    >>> # 简单创建
    >>> dataset = create_dataset(
    ...     "era5",
    ...     path="/data/era5",
    ...     split="train",
    ...     variables=["t2m", "u10", "v10"],
    ... )
    
    >>> # 使用配置文件
    >>> dataset = create_dataset("era5", config="config.yaml")
    
    >>> # 使用配置字典
    >>> config = {
    ...     "source": {"path": "/data/era5", "split": "train"},
    ...     "data": {"variables": ["t2m", "u10", "v10"]},
    ... }
    >>> dataset = create_dataset("era5", config=config)
    """
    return DatasetFactory.create(name, config, **kwargs)


def create_dataset_from_config(
    config_path: Union[str, Path],
    **kwargs
) -> BaseDataset:
    """
    从配置文件创建数据集（便捷函数）
    
    Parameters
    ----------
    config_path : Union[str, Path]
        配置文件路径
    **kwargs
        额外参数
        
    Returns
    -------
    BaseDataset
        数据集实例
    """
    return DatasetFactory.create_from_config_file(config_path, **kwargs)


def create_datasets(
    configs: list[Union[str, Dict[str, Any], DatasetConfig]],
    **common_kwargs
) -> list[BaseDataset]:
    """
    批量创建数据集（便捷函数）
    
    Parameters
    ----------
    configs : List[Union[str, Dict, DatasetConfig]]
        数据集配置列表
    **common_kwargs
        所有数据集共享的参数
        
    Returns
    -------
    List[BaseDataset]
        数据集实例列表
    """
    return DatasetFactory.create_multi(configs, **common_kwargs)
