from torch import nn
from onescience.modules._lazy import instantiate_registered_style

# 构建统一的注册表
_HEAD_REGISTRY = {
    "MaskedMSAHead": ("onescience.modules.head.maskedmsahead", "MaskedMSAHead"),
    "UNetHead1D": ("onescience.modules.head.unet_head", "UNetHead1D"),
    "UNetHead2D": ("onescience.modules.head.unet_head", "UNetHead2D"),
    "UNetHead3D": ("onescience.modules.head.unet_head", "UNetHead3D"),
    "EnergyHead": ("onescience.modules.head.matris_head", "EnergyHead"),
    "MagmomHead": ("onescience.modules.head.matris_head", "MagmomHead"),
    "ForceStressHead": ("onescience.modules.head.matris_head", "ForceStressHead"),
}

class OneHead(nn.Module):
    """
    OneHead 统一预测头调用接口。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.head = instantiate_registered_style(style, _HEAD_REGISTRY, "head", **kwargs)

    def forward(self, *args, **kwargs):
        """
        前向传播
        """
        return self.head(*args, **kwargs)
