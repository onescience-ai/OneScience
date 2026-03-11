import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp
from onescience.modules.attention.oneattention import OneAttention
from onescience.modules.mlp.onemlp import OneMlp


class XiheGlobalSIEFuser(nn.Module):
    """
    Xihe 模型的全局特征融合模块，基于分组注意力机制实现全局特征交互。

    Args:
        dim (int): 输入通道数 C。
        num_heads (int): 注意力头数量，默认为 12。
        qkv_bias (bool): 是否在 QKV 上添加可学习的偏置，默认为 True。
        num_groups (int): 特征分组数量 G，默认为 32。
        norm_layer (nn.Module): 归一化层类型，默认为 nn.LayerNorm。

    形状:
        输入 obj: 包含以下属性的对象
            - x (torch.Tensor): 输入张量，形状为 (B, L, C)，其中 L = Pl × Lat × Lon
            - mask (torch.Tensor, optional): 掩码张量，形状为 (B, L) 或可 reshape 为 (B, L)
            - y (torch.Tensor, optional): 用于残差连接的辅助张量，形状与 x 相同
        输出 x (torch.Tensor): 输出张量，形状与输入 x 相同，为 (B, L, C)

    Example:
        >>> global_fuser = XiheGlobalSIEFuser(
        ...     dim=192,
        ...     num_heads=12,
        ...     num_groups=32
        ... )
        >>> from types import SimpleNamespace
        >>> B, L, C = 2, 425984, 192  # L = 13*128*256
        >>> x = torch.randn(B, L, C)
        >>> mask = torch.ones(B, L, dtype=torch.bool)
        >>> obj = SimpleNamespace(x=x, mask=mask)
        >>> out = global_fuser(obj)
        >>> out.shape
        torch.Size([2, 425984, 192])
    """
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=True,
        num_groups=32,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim=dim
        self.feature_grouping=OneAttention(dim=dim,style="FeatureGroupingAttention")
        self.feature_ungrouping=OneAttention(dim=dim,style="FeatureUngroupingAttention")
        
        self.group_propagation = OneMlp(dim=dim,style="XiheMlp") 
    def forward(self, obj):
        x=obj.x
        mask=obj.mask
        obj.y=x
        
        x=self.feature_grouping(obj,mask=obj.mask)
        x=self.group_propagation(x)
        obj.x=x
        x=self.feature_ungrouping(obj,mask=obj.mask)
 
        return x
    