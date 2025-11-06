"""
数据集注册中心

管理所有可用的数据集类
"""

from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
import logging

from onescience.datapipes.core.base_dataset import BaseDataset


logger = logging.getLogger("onescience.datapipes.registry")


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    cls: Type[BaseDataset]
    domain: str
    task: Optional[str] = None
    description: Optional[str] = None
    required_params: Optional[List[str]] = None
    optional_params: Optional[List[str]] = None
    data_formats: Optional[List[str]] = None


class DatasetRegistry:
    """
    数据集注册中心
    
    管理所有可用的数据集类，提供注册、查询和发现功能
    """
    
    _registry: Dict[str, DatasetInfo] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        dataset_cls: Type[BaseDataset],
        domain: str,
        task: Optional[str] = None,
        description: Optional[str] = None,
        required_params: Optional[List[str]] = None,
        optional_params: Optional[List[str]] = None,
        data_formats: Optional[List[str]] = None,
    ):
        """
        注册数据集
        
        Parameters
        ----------
        name : str
            数据集名称（唯一标识符）
        dataset_cls : Type[BaseDataset]
            数据集类
        domain : str
            领域（earth, cfd, biology, materials, structural）
        task : Optional[str]
            任务类型
        description : Optional[str]
            数据集描述
        required_params : Optional[List[str]]
            必需参数列表
        optional_params : Optional[List[str]]
            可选参数列表
        data_formats : Optional[List[str]]
            支持的数据格式列表
        """
        if not issubclass(dataset_cls, BaseDataset):
            raise TypeError(f"{dataset_cls.__name__} must inherit from BaseDataset")
        
        if name in cls._registry:
            logger.warning(f"Dataset '{name}' is already registered. Overwriting...")
        
        # 从类属性获取默认值
        if required_params is None and hasattr(dataset_cls, 'REQUIRED_PARAMS'):
            required_params = dataset_cls.REQUIRED_PARAMS
        
        if optional_params is None and hasattr(dataset_cls, 'OPTIONAL_PARAMS'):
            optional_params = dataset_cls.OPTIONAL_PARAMS
        
        if data_formats is None and hasattr(dataset_cls, 'DATA_FORMATS'):
            data_formats = dataset_cls.DATA_FORMATS
        
        info = DatasetInfo(
            name=name,
            cls=dataset_cls,
            domain=domain,
            task=task,
            description=description,
            required_params=required_params,
            optional_params=optional_params,
            data_formats=data_formats,
        )
        
        cls._registry[name] = info
        logger.info(f"Registered dataset: {name} (domain={domain}, task={task})")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseDataset]]:
        """
        获取数据集类
        
        Parameters
        ----------
        name : str
            数据集名称
            
        Returns
        -------
        Optional[Type[BaseDataset]]
            数据集类，如果不存在则返回None
        """
        info = cls._registry.get(name)
        return info.cls if info else None
    
    @classmethod
    def get_info(cls, name: str) -> Optional[DatasetInfo]:
        """
        获取数据集信息
        
        Parameters
        ----------
        name : str
            数据集名称
            
        Returns
        -------
        Optional[DatasetInfo]
            数据集信息
        """
        return cls._registry.get(name)
    
    @classmethod
    def list(cls) -> List[str]:
        """
        列出所有注册的数据集名称
        
        Returns
        -------
        List[str]
            数据集名称列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def list_all(cls) -> Dict[str, DatasetInfo]:
        """
        列出所有注册的数据集及其信息
        
        Returns
        -------
        Dict[str, DatasetInfo]
            数据集信息字典
        """
        return cls._registry.copy()
    
    @classmethod
    def search(
        cls,
        domain: Optional[str] = None,
        task: Optional[str] = None,
        data_format: Optional[str] = None,
    ) -> List[str]:
        """
        搜索数据集
        
        Parameters
        ----------
        domain : Optional[str]
            领域过滤
        task : Optional[str]
            任务类型过滤
        data_format : Optional[str]
            数据格式过滤
            
        Returns
        -------
        List[str]
            符合条件的数据集名称列表
        """
        results = []
        
        for name, info in cls._registry.items():
            # 领域过滤
            if domain is not None and info.domain != domain:
                continue
            
            # 任务过滤
            if task is not None and info.task != task:
                continue
            
            # 数据格式过滤
            if data_format is not None:
                if not info.data_formats or data_format not in info.data_formats:
                    continue
            
            results.append(name)
        
        return results
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """
        检查数据集是否存在
        
        Parameters
        ----------
        name : str
            数据集名称
            
        Returns
        -------
        bool
            是否存在
        """
        return name in cls._registry
    
    @classmethod
    def unregister(cls, name: str):
        """
        注销数据集
        
        Parameters
        ----------
        name : str
            数据集名称
        """
        if name in cls._registry:
            del cls._registry[name]
            logger.info(f"Unregistered dataset: {name}")
        else:
            logger.warning(f"Dataset '{name}' not found in registry")
    
    @classmethod
    def clear(cls):
        """清空注册表"""
        cls._registry.clear()
        logger.info("Cleared dataset registry")
    
    @classmethod
    def print_registry(cls):
        """打印注册表"""
        print("\n" + "=" * 80)
        print(f"{'Dataset Registry':^80}")
        print("=" * 80)
        print(f"Total datasets: {len(cls._registry)}\n")
        
        # 按领域分组
        domains = {}
        for name, info in cls._registry.items():
            if info.domain not in domains:
                domains[info.domain] = []
            domains[info.domain].append((name, info))
        
        for domain, datasets in sorted(domains.items()):
            print(f"\n{domain.upper()}")
            print("-" * 80)
            for name, info in sorted(datasets, key=lambda x: x[0]):
                task_str = f" (task={info.task})" if info.task else ""
                print(f"  • {name}{task_str}")
                if info.description:
                    print(f"    {info.description}")
                if info.data_formats:
                    print(f"    Formats: {', '.join(info.data_formats)}")
        
        print("\n" + "=" * 80 + "\n")


def register_dataset(
    name: str,
    domain: Optional[str] = None,
    task: Optional[str] = None,
    description: Optional[str] = None,
    required_params: Optional[List[str]] = None,
    optional_params: Optional[List[str]] = None,
    data_formats: Optional[List[str]] = None,
):
    """
    数据集注册装饰器
    
    Parameters
    ----------
    name : str
        数据集名称
    domain : Optional[str]
        领域
    task : Optional[str]
        任务类型
    description : Optional[str]
        描述
    required_params : Optional[List[str]]
        必需参数
    optional_params : Optional[List[str]]
        可选参数
    data_formats : Optional[List[str]]
        支持的数据格式
        
    Examples
    --------
    >>> @register_dataset(name="MyDataset", domain="earth", task="forecasting")
    >>> class MyDataset(EarthDataset):
    >>>     pass
    """
    def decorator(cls: Type[BaseDataset]):
        # 从类属性获取领域（如果未指定）
        _domain = domain
        if _domain is None and hasattr(cls, 'DOMAIN'):
            _domain = cls.DOMAIN
        
        if _domain is None:
            raise ValueError("Domain must be specified either in decorator or class attribute")
        
        DatasetRegistry.register(
            name=name,
            dataset_cls=cls,
            domain=_domain,
            task=task,
            description=description,
            required_params=required_params,
            optional_params=optional_params,
            data_formats=data_formats,
        )
        return cls
    return decorator


def get_dataset(name: str) -> Optional[Type[BaseDataset]]:
    """
    获取数据集类（便捷函数）
    
    Parameters
    ----------
    name : str
        数据集名称
        
    Returns
    -------
    Optional[Type[BaseDataset]]
        数据集类
    """
    return DatasetRegistry.get(name)


def list_datasets() -> List[str]:
    """
    列出所有数据集（便捷函数）
    
    Returns
    -------
    List[str]
        数据集名称列表
    """
    return DatasetRegistry.list()


def search_datasets(
    domain: Optional[str] = None,
    task: Optional[str] = None,
    data_format: Optional[str] = None,
) -> List[str]:
    """
    搜索数据集（便捷函数）
    
    Parameters
    ----------
    domain : Optional[str]
        领域过滤
    task : Optional[str]
        任务类型过滤
    data_format : Optional[str]
        数据格式过滤
        
    Returns
    -------
    List[str]
        符合条件的数据集名称列表
    """
    return DatasetRegistry.search(domain=domain, task=task, data_format=data_format)

