from torch import nn

from .pangupatchrecovery import PanguPatchRecovery
from .xihepatchrecovery    import XihePatchRecovery

_RECOVERY_REGISTRY = {
    "PanguPatchRecovery": PanguPatchRecovery,
    "XihePatchRecovery":XihePatchRecovery,
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

        if style not in _RECOVERY_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.recovery = _RECOVERY_REGISTRY[style](**kwargs)
        self.Reconvery = self.recovery

    def forward(self, *args, **kwargs):
        return self.recovery(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.recovery, name)
