from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_AFNO_REGISTRY = {
    "FourCastNetAFNO2D": (
        "onescience.modules.afno.fourcastnetafno",
        "FourCastNetAFNO2D",
    ),
}

class OneAFNO(nn.Module):
    """
    AFNO 模块统一入口。

    通过 `style` 从注册表中选择具体 AFNO 实现。
    当前天气相关模型中，常用实现包括：

    - `FourCastNetAFNO2D`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.afno = instantiate_registered_style(style, _AFNO_REGISTRY, "AFNO", **kwargs)

    def forward(self, x):
        return self.afno(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.afno, name)
