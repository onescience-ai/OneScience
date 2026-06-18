from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_SAMPLER_REGISTRY = {
    "PanguDownSample": ("onescience.modules.sample.pangudownsample", "PanguDownSample"),
    "PanguUpSample": ("onescience.modules.sample.panguupsample", "PanguUpSample"),
    "SpatialGraphDownsample": (
        "onescience.modules.sample.SpatialGraphDownsample",
        "SpatialGraphDownsample",
    ),
    "SpatialGraphUpsample": (
        "onescience.modules.sample.SpatialGraphUpsample",
        "SpatialGraphUpsample",
    ),
    "FuxiUpSample": ("onescience.modules.sample.fuxiupsample", "FuxiUpSample"),
    "FuxiDownSample": ("onescience.modules.sample.fuxidownsample", "FuxiDownSample"),
    "XiheUpSample": ("onescience.modules.sample.xiheupsample", "XiheUpSample"),
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

        self.sampler = instantiate_registered_style(style, _SAMPLER_REGISTRY, "sample", **kwargs)
        self.Sampler = self.sampler

    def forward(self, *args, **kwargs):
        return self.sampler(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.sampler, name)
