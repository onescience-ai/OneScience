from torch import nn

from .fuxifc import FuxiFC
from .fourcastnetfc import FourCastNetFC

_FC_REGISTRY = {
    "FuxiFC": FuxiFC,
    "FourCastNetFC": FourCastNetFC,
}

class OneFC(nn.Module):
    """
    全连接模块统一入口。

    通过 `style` 从注册表中选择具体逐位置前馈实现。
    当前天气相关模型中，常用实现包括：

    - `FourCastNetFC`
    - `FuxiFC`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _FC_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.fc = _FC_REGISTRY[style](**kwargs)
        
    def forward(self, *args, **kwargs):
        return self.fc(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.fc, name)
