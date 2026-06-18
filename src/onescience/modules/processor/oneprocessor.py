import torch.nn as nn

from onescience.modules._lazy import instantiate_registered_style

_PROCESSOR_REGISTRY = {
    "BistrideGraphMessagePassing": (
        "onescience.modules.processor.bistride_processor",
        "BistrideGraphMessagePassing",
    ),
    "GraphMessagePassing": (
        "onescience.modules.processor.bistride_processor",
        "GraphMessagePassing",
    ),
}

class OneProcessor(nn.Module):
    """
    OneProcessor: 统一处理器调用接口。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()
        self.processor = instantiate_registered_style(
            style,
            _PROCESSOR_REGISTRY,
            "processor",
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        return self.processor(*args, **kwargs)
