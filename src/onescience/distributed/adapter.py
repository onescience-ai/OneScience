"""
分布式环境适配器模块

提供统一的分布式操作接口，自动检测并适配不同的分布式环境。

支持的分布式环境：
- Megatron-LM: mpu.is_initialized()
- DistributedManager: DistributedManager._is_initialized
- PyTorch 原生: torch.distributed.is_initialized()
- 单机: 以上均未初始化
"""

from typing import Callable, Optional


def detect_distributed_environment() -> str:
    """
    检测当前运行的分布式环境
    
    Returns:
        str: 环境类型，可选值：
            - 'megatron': Megatron-LM 环境
            - 'distributed_manager': DistributedManager 环境
            - 'torch_distributed': PyTorch 原生分布式环境
            - 'single': 单机环境
    """
    # 1. 检测 Megatron-LM
    try:
        from onescience.distributed.megatron.core import mpu
        if mpu.is_initialized():
            return 'megatron'
    except (ImportError, AttributeError):
        pass
    
    # 2. 检测 DistributedManager
    try:
        from onescience.distributed.manager import DistributedManager
        if DistributedManager._is_initialized:
            return 'distributed_manager'
    except (ImportError, AttributeError):
        pass
    
    # 3. 检测 PyTorch 原生分布式
    try:
        import torch
        if torch.distributed.is_initialized():
            return 'torch_distributed'
    except (ImportError, AttributeError):
        pass
    
    # 4. 单机环境
    return 'single'


def get_rank0_checker(env_type: str) -> Callable[[], bool]:
    """
    获取 rank0 检查函数
    
    Args:
        env_type: 环境类型
        
    Returns:
        Callable[[], bool]: rank0 检查函数
    """
    if env_type == 'megatron':
        from onescience.distributed.megatron.core import mpu
        return lambda: mpu.get_data_parallel_rank() == 0
    elif env_type == 'distributed_manager':
        from onescience.distributed.manager import DistributedManager
        dist = DistributedManager()
        return lambda: dist.rank == 0
    elif env_type == 'torch_distributed':
        import torch
        return lambda: torch.distributed.get_rank() == 0
    else:  # 'single'
        return lambda: True


def get_rank(env_type: str) -> int:
    """
    获取当前进程的 rank
    
    Args:
        env_type: 环境类型
        
    Returns:
        int: 当前进程的 rank
    """
    if env_type == 'megatron':
        from onescience.distributed.megatron.core import mpu
        return mpu.get_data_parallel_rank()
    elif env_type == 'distributed_manager':
        from onescience.distributed.manager import DistributedManager
        dist = DistributedManager()
        return dist.rank
    elif env_type == 'torch_distributed':
        import torch
        return torch.distributed.get_rank()
    else:  # 'single'
        return 0


def get_world_size(env_type: str) -> int:
    """
    获取分布式环境的世界大小
    
    Args:
        env_type: 环境类型
        
    Returns:
        int: 世界大小
    """
    if env_type == 'megatron':
        from onescience.distributed.megatron.core import mpu
        return mpu.get_data_parallel_world_size()
    elif env_type == 'distributed_manager':
        from onescience.distributed.manager import DistributedManager
        dist = DistributedManager()
        return dist.world_size
    elif env_type == 'torch_distributed':
        import torch
        return torch.distributed.get_world_size()
    else:  # 'single'
        return 1


def barrier(env_type: str) -> None:
    """
    同步所有进程
    
    Args:
        env_type: 环境类型
    """
    if env_type == 'megatron':
        import torch
        torch.distributed.barrier()
    elif env_type == 'distributed_manager':
        from onescience.distributed.manager import DistributedManager
        dist = DistributedManager()
        dist.barrier()
    elif env_type == 'torch_distributed':
        import torch
        torch.distributed.barrier()
    # 'single' 环境不需要 barrier


class DistributedAdapter:
    """
    分布式环境适配器类
    
    提供统一的分布式操作接口，自动检测并适配不同的分布式环境。
    """
    
    def __init__(self):
        """
        初始化适配器，自动检测分布式环境
        """
        self._env_type = detect_distributed_environment()
        self._is_rank0_func = get_rank0_checker(self._env_type)
    
    @property
    def env_type(self) -> str:
        """
        获取当前环境类型
        
        Returns:
            str: 环境类型
        """
        return self._env_type
    
    def is_rank0(self) -> bool:
        """
        判断当前进程是否为 rank 0
        
        Returns:
            bool: 如果是 rank 0 返回 True，否则返回 False
        """
        return self._is_rank0_func()
    
    def get_rank(self) -> int:
        """
        获取当前进程的 rank
        
        Returns:
            int: 当前进程的 rank
        """
        return get_rank(self._env_type)
    
    def get_world_size(self) -> int:
        """
        获取分布式环境的世界大小
        
        Returns:
            int: 世界大小
        """
        return get_world_size(self._env_type)
    
    def barrier(self) -> None:
        """
        同步所有进程
        """
        barrier(self._env_type)


def create_adapter() -> DistributedAdapter:
    """
    创建分布式适配器实例（便捷函数）
    
    Returns:
        DistributedAdapter: 分布式适配器实例
    """
    return DistributedAdapter()