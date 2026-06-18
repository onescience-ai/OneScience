import torch.nn as nn
from onescience.modules._lazy import instantiate_registered_style

# 构建统一的注册表
_EQUIVARIANT_REGISTRY = {
    # 标准等变卷积
    "GroupEquivariantConv2d": (
        "onescience.modules.equivariant.group_conv",
        "GroupEquivariantConv2d",
    ),
    "GroupEquivariantConv3d": (
        "onescience.modules.equivariant.group_conv",
        "GroupEquivariantConv3d",
    ),
}

class OneEquivariant(nn.Module):
    """
    OneEquivariant: 统一等变层调用接口。
    
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.equivariant_layer = instantiate_registered_style(
            style,
            _EQUIVARIANT_REGISTRY,
            "equivariant",
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        """
        前向传播。
        """
        return self.equivariant_layer(*args, **kwargs)
