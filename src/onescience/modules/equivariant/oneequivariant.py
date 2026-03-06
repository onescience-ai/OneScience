import torch.nn as nn
from .group_conv import GroupEquivariantConv2d, GroupEquivariantConv3d

# 构建统一的注册表
_EQUIVARIANT_REGISTRY = {
    # 标准等变卷积
    "GroupEquivariantConv2d": GroupEquivariantConv2d,
    "GroupEquivariantConv3d": GroupEquivariantConv3d,
}

class OneEquivariant(nn.Module):
    """
    OneEquivariant: 统一等变层调用接口。
    
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _EQUIVARIANT_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_EQUIVARIANT_REGISTRY.keys())}"
            )
        
        # 实例化具体的等变层
        self.equivariant_layer = _EQUIVARIANT_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        """
        前向传播。
        """
        return self.equivariant_layer(*args, **kwargs)