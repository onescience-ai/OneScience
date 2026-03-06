import torch.nn as nn

# 导入具体的 Processor 实现
from .bistride_processor import BistrideGraphMessagePassing, GraphMessagePassing

_PROCESSOR_REGISTRY = {
    "BistrideGraphMessagePassing": BistrideGraphMessagePassing,
    "GraphMessagePassing": GraphMessagePassing,
}

class OneProcessor(nn.Module):
    """
    OneProcessor: 统一处理器调用接口。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()
        if style not in _PROCESSOR_REGISTRY:
            raise NotImplementedError(
                f"Unknown processor style: '{style}'. Available: {list(_PROCESSOR_REGISTRY.keys())}"
            )
        
        self.processor = _PROCESSOR_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.processor(*args, **kwargs)