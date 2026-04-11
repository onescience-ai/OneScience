from torch import nn

from .pangudownsample import PanguDownSample
from .panguupsample import PanguUpSample
from .SpatialGraphDownsample import SpatialGraphDownsample
from .SpatialGraphUpsample import SpatialGraphUpsample
from .fuxidownsample import FuxiDownSample
from .fuxiupsample import FuxiUpSample
from .xiheupsample import XiheUpSample

_SAMPLER_REGISTRY = {
    "PanguDownSample": PanguDownSample,
    "PanguUpSample": PanguUpSample,
    "SpatialGraphDownsample": SpatialGraphDownsample,
    "SpatialGraphUpsample": SpatialGraphUpsample,
    "FuxiUpSample": FuxiUpSample,
    "FuxiDownSample": FuxiDownSample,
    "XiheUpSample": XiheUpSample,
}


class OneSample(nn.Module):
    """
    采样模块统一入口。

    通过 `style` 从注册表中选择具体下采样或上采样实现。
    当前天气相关模型中，常用实现包括：

    - `PanguDownSample`
    - `PanguUpSample`
    - `FuxiDownSample`
    - `FuxiUpSample`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _SAMPLER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.sampler = _SAMPLER_REGISTRY[style](**kwargs)
        self.Sampler = self.sampler

    def forward(self, *args, **kwargs):
        return self.sampler(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.sampler, name)
