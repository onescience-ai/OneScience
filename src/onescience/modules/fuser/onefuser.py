from torch import nn

from .pangufuser import PanguFuser
from .fengwufuser import FengWuFuser
from .fourcastnetfuser import FourCastNetFuser
from .xihelocalsiefuser import XiheLocalSIEFuser
from .xiheglobalsiefuser import XiheGlobalSIEFuser
from .xihefuse import XiheFuser

_FUSER_REGISTRY = {
    "PanguFuser": PanguFuser,
    "FengWuFuser": FengWuFuser,
    "FourCastNetFuser": FourCastNetFuser,
    "XiheLocalSIEFuser":XiheLocalSIEFuser,
    "XiheGlobalSIEFuser":XiheGlobalSIEFuser,
    "XiheFuser":XiheFuser,
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

        if style not in _FUSER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.fuser = _FUSER_REGISTRY[style](**kwargs)
        self.Fuser = self.fuser

    def forward(self, *args, **kwargs):
        return self.fuser(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.fuser, name)
