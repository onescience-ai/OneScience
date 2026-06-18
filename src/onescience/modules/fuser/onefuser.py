from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_FUSER_REGISTRY = {
    "PanguFuser": ("onescience.modules.fuser.pangufuser", "PanguFuser"),
    "FengWuFuser": ("onescience.modules.fuser.fengwufuser", "FengWuFuser"),
    "FourCastNetFuser": ("onescience.modules.fuser.fourcastnetfuser", "FourCastNetFuser"),
    "XiheLocalSIEFuser": ("onescience.modules.fuser.xihelocalsiefuser", "XiheLocalSIEFuser"),
    "XiheGlobalSIEFuser": ("onescience.modules.fuser.xiheglobalsiefuser", "XiheGlobalSIEFuser"),
    "XiheFuser": ("onescience.modules.fuser.xihefuse", "XiheFuser"),
}

class OneFuser(nn.Module):
    """
    融合模块统一入口。

    通过 `style` 从注册表中选择具体融合实现。
    当前天气相关模型中，常用实现包括：

    - `PanguFuser`
    - `FengWuFuser`
    - `FourCastNetFuser`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.fuser = instantiate_registered_style(style, _FUSER_REGISTRY, "fuser", **kwargs)
        self.Fuser = self.fuser

    def forward(self, *args, **kwargs):
        return self.fuser(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.fuser, name)
