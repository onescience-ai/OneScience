import torch
from torch import nn
from .maskedmsahead import MaskedMSAHead
from .unet_head import UNetHead1D, UNetHead2D, UNetHead3D

try:
    from .mace_readout_blocks import (
        LinearDipoleReadoutBlock as MaceLinearDipoleReadoutBlock,
        LinearReadoutBlock as MaceLinearReadoutBlock,
        NonLinearDipoleReadoutBlock as MaceNonLinearDipoleReadoutBlock,
        NonLinearReadoutBlock as MaceNonLinearReadoutBlock,
        ScaleShiftBlock as MaceScaleShiftBlock,
    )
except Exception:  # pragma: no cover - optional MACE deps
    MaceLinearDipoleReadoutBlock = None
    MaceLinearReadoutBlock = None
    MaceNonLinearDipoleReadoutBlock = None
    MaceNonLinearReadoutBlock = None
    MaceScaleShiftBlock = None

# 构建统一的注册表
_HEAD_REGISTRY = {
    "MaskedMSAHead": MaskedMSAHead,
    "UNetHead1D": UNetHead1D,
    "UNetHead2D": UNetHead2D,
    "UNetHead3D": UNetHead3D,
}

if MaceLinearReadoutBlock is not None:
    _HEAD_REGISTRY.update(
        {
            "MaceLinearReadoutBlock": MaceLinearReadoutBlock,
            "MaceNonLinearReadoutBlock": MaceNonLinearReadoutBlock,
            "MaceLinearDipoleReadoutBlock": MaceLinearDipoleReadoutBlock,
            "MaceNonLinearDipoleReadoutBlock": MaceNonLinearDipoleReadoutBlock,
            "MaceScaleShiftBlock": MaceScaleShiftBlock,
        }
    )

class OneHead(nn.Module):
    """
    OneHead 统一预测头调用接口。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _HEAD_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_HEAD_REGISTRY.keys())}"
            )
        
        # 实例化具体的预测头层
        self.head = _HEAD_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        """
        前向传播
        """
        return self.head(*args, **kwargs)

