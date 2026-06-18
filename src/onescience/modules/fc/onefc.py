from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_FC_REGISTRY = {
    "FuxiFC": ("onescience.modules.fc.fuxifc", "FuxiFC"),
    "FourCastNetFC": ("onescience.modules.fc.fourcastnetfc", "FourCastNetFC"),
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

        self.fc = instantiate_registered_style(style, _FC_REGISTRY, "fc", **kwargs)
        
    def forward(self, *args, **kwargs):
        return self.fc(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.fc, name)
