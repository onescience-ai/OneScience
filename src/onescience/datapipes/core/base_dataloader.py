"""
BaseDataLoader - 统一的DataLoader创建工厂

提供标准化的DataLoader配置和创建
"""

import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .config import DataLoaderConfig


logger = logging.getLogger("onescience.datapipes.dataloader")


class BaseDataLoader:
    """
    DataLoader工厂类
    
    提供统一的DataLoader创建接口
    """
    
    @staticmethod
    def create(
        dataset: Dataset,
        config: Union[DataLoaderConfig, Dict[str, Any]],
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """
        创建DataLoader
        
        Parameters
        ----------
        dataset : Dataset
            数据集
        config : Union[DataLoaderConfig, Dict[str, Any]]
            DataLoader配置
        collate_fn : Optional[Callable]
            自定义collate函数
            
        Returns
        -------
        DataLoader
            配置好的DataLoader
        """
        # 配置处理
        if isinstance(config, dict):
            config = DataLoaderConfig.from_dict(config)
        
        # Sampler配置
        sampler = None
        shuffle = config.shuffle
        
        if config.distributed.enabled:
            sampler = DistributedSampler(
                dataset,
                num_replicas=config.distributed.world_size,
                rank=config.distributed.rank,
                shuffle=config.distributed.shuffle,
                seed=config.distributed.seed,
                drop_last=config.distributed.drop_last,
            )
            shuffle = False  # 使用sampler时不能shuffle
            logger.info(
                f"Created DistributedSampler with rank={config.distributed.rank}, "
                f"world_size={config.distributed.world_size}"
            )
        
        # Collate函数
        if collate_fn is None and config.collate_fn:
            collate_fn = get_collate_fn(config.collate_fn)
        
        # 创建DataLoader
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": config.batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "drop_last": config.drop_last,
        }
        
        if collate_fn is not None:
            dataloader_kwargs["collate_fn"] = collate_fn
        
        # 添加可选参数
        if config.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = config.prefetch_factor
            dataloader_kwargs["persistent_workers"] = config.persistent_workers
        
        dataloader = DataLoader(**dataloader_kwargs)
        
        logger.info(
            f"Created DataLoader with batch_size={config.batch_size}, "
            f"num_workers={config.num_workers}"
        )
        
        return dataloader
    
    @staticmethod
    def create_train_dataloader(
        dataset: Dataset,
        config: Union[DataLoaderConfig, Dict[str, Any]],
        **kwargs
    ) -> DataLoader:
        """
        创建训练DataLoader
        
        自动设置适合训练的配置（shuffle=True等）
        """
        if isinstance(config, dict):
            config = DataLoaderConfig.from_dict(config)
        
        # 训练时默认shuffle
        if not config.distributed.enabled:
            config.shuffle = True
        
        config.drop_last = True
        
        return BaseDataLoader.create(dataset, config, **kwargs)
    
    @staticmethod
    def create_val_dataloader(
        dataset: Dataset,
        config: Union[DataLoaderConfig, Dict[str, Any]],
        **kwargs
    ) -> DataLoader:
        """
        创建验证DataLoader
        
        自动设置适合验证的配置（shuffle=False等）
        """
        if isinstance(config, dict):
            config = DataLoaderConfig.from_dict(config)
        
        # 验证时不shuffle
        config.shuffle = False
        config.drop_last = False
        
        return BaseDataLoader.create(dataset, config, **kwargs)
    
    @staticmethod
    def create_test_dataloader(
        dataset: Dataset,
        config: Union[DataLoaderConfig, Dict[str, Any]],
        **kwargs
    ) -> DataLoader:
        """
        创建测试DataLoader
        
        自动设置适合测试的配置（shuffle=False, drop_last=False等）
        """
        if isinstance(config, dict):
            config = DataLoaderConfig.from_dict(config)
        
        # 测试时不shuffle，不drop_last
        config.shuffle = False
        config.drop_last = False
        
        # 测试时不使用分布式
        config.distributed.enabled = False
        
        return BaseDataLoader.create(dataset, config, **kwargs)


def create_dataloader(
    dataset: Dataset,
    config: Union[DataLoaderConfig, Dict[str, Any]],
    split: str = "train",
    **kwargs
) -> DataLoader:
    """
    便捷函数：根据split自动创建合适的DataLoader
    
    Parameters
    ----------
    dataset : Dataset
        数据集
    config : Union[DataLoaderConfig, Dict[str, Any]]
        配置
    split : str
        数据集分割 (train, val, test)
    **kwargs
        其他参数
        
    Returns
    -------
    DataLoader
        DataLoader实例
    """
    if split == "train":
        return BaseDataLoader.create_train_dataloader(dataset, config, **kwargs)
    elif split == "val" or split == "valid":
        return BaseDataLoader.create_val_dataloader(dataset, config, **kwargs)
    elif split == "test":
        return BaseDataLoader.create_test_dataloader(dataset, config, **kwargs)
    else:
        return BaseDataLoader.create(dataset, config, **kwargs)


# ============================================================================
# Collate函数注册
# ============================================================================

_COLLATE_FN_REGISTRY: Dict[str, Callable] = {}


def register_collate_fn(name: str):
    """
    注册collate函数
    
    Parameters
    ----------
    name : str
        collate函数名称
    """
    def decorator(func: Callable):
        _COLLATE_FN_REGISTRY[name] = func
        return func
    return decorator


def get_collate_fn(name: str) -> Optional[Callable]:
    """
    获取注册的collate函数
    
    Parameters
    ----------
    name : str
        collate函数名称
        
    Returns
    -------
    Optional[Callable]
        collate函数
    """
    if name == "default":
        return None  # 使用PyTorch默认的collate
    return _COLLATE_FN_REGISTRY.get(name)


def list_collate_fns() -> Dict[str, Callable]:
    """列出所有注册的collate函数"""
    return _COLLATE_FN_REGISTRY.copy()


# ============================================================================
# 内置Collate函数
# ============================================================================

@register_collate_fn("dict_collate")
def dict_collate(batch):
    """
    字典数据的collate函数
    
    将列表的字典转换为字典的张量
    """
    if not batch:
        return {}
    
    # 假设batch是一个字典列表
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # 如果是tensor或numpy数组，stack它们
        if isinstance(values[0], (torch.Tensor, int, float)):
            result[key] = torch.stack([
                v if isinstance(v, torch.Tensor) else torch.tensor(v)
                for v in values
            ])
        else:
            # 否则保持为列表
            result[key] = values
    
    return result


@register_collate_fn("graph_collate")
def graph_collate(batch):
    """
    图数据的collate函数
    
    用于DGL或PyG图数据
    """
    try:
        import dgl
        # 如果是DGL图
        if isinstance(batch[0], dgl.DGLGraph):
            return dgl.batch(batch)
    except ImportError:
        pass
    
    try:
        from torch_geometric.data import Batch
        # 如果是PyG图
        return Batch.from_data_list(batch)
    except ImportError:
        pass
    
    raise ImportError("Neither DGL nor PyTorch Geometric is installed")

