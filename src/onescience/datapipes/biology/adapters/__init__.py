"""
模型适配器层

将通用处理模块的输出转换为各模型需要的特定格式
"""

from onescience.datapipes.biology.adapters.base_adapter import BaseAdapter
from onescience.datapipes.biology.adapters.adapter_registry import (
    get_adapter,
    register_adapter,
)

try:
    from onescience.datapipes.biology.adapters.protenix_infer_adapter import ProtenixInferAdapter
except ImportError:
    ProtenixAdapter = None  # type: ignore
from onescience.datapipes.biology.adapters.adapter_registry import (
    get_adapter,
    register_adapter,
)

__all__ = [
    "BaseAdapter",
    "get_adapter",
    "register_adapter",
    "ProtenixInferAdapter",
]

