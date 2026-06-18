from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_RECOVERY_REGISTRY = {
    "PanguPatchRecovery": (
        "onescience.modules.recovery.pangupatchrecovery",
        "PanguPatchRecovery",
    ),
    "XihePatchRecovery": (
        "onescience.modules.recovery.xihepatchrecovery",
        "XihePatchRecovery",
    ),
}

class OneRecovery(nn.Module):
    """
    恢复模块统一入口。

    通过 `style` 从注册表中选择具体恢复实现。
    当前天气相关模型中，常用实现包括：

    - `PanguPatchRecovery`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.recovery = instantiate_registered_style(
            style,
            _RECOVERY_REGISTRY,
            "recovery",
            **kwargs,
        )
        self.Reconvery = self.recovery

    def forward(self, *args, **kwargs):
        return self.recovery(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.recovery, name)
