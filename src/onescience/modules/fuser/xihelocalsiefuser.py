from collections.abc import Sequence
import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp,crop3d,get_pad3d,window_partition,window_reverse,DropPath
from onescience.modules.attention.oneattention import OneAttention
from onescience.modules.transformer.onetransformer import OneTransformer

class XiheLocalSIEFuser(nn.Module):
    """
    基于 WeatherLearn 的 3D 局部 SIEFuser 模块（用于 Xihe 模型的多层局部特征融合）。

    Args:
        dim (int): 输入通道数 C。
        input_resolution (tuple[int, int, int]): 输入空间分辨率 (Pl, Lat, Lon)。
        depth (int): Transformer 块的数量，默认为 2。
        num_heads (int): 注意力头数量，默认为 6。
        window_size (tuple[int, int, int]): 3D 局部窗口大小 (Wpl, Wlat, Wlon)，默认为 (1, 6, 12)。
        mlp_ratio (float): MLP 隐层扩展比例，默认为 4.0。
        qkv_bias (bool): 是否在 QKV 上添加可学习的偏置，默认为 True。
        qk_scale (float | None): 覆盖默认 QK 缩放系数 (head_dim ** -0.5)，默认为 None。
        drop (float): 输出/MLP dropout 比例，默认为 0.0。
        attn_drop (float): 注意力权重 dropout 比例，默认为 0.0。
        drop_path (float | tuple[float]): 随机深度（DropPath）比例，默认为 0.0。
        norm_layer (nn.Module): 归一化层类型，默认为 nn.LayerNorm。

    形状:
        输入 obj: 包含以下属性的对象
            - x (torch.Tensor): 输入张量，形状为 (B, L, C) 或 (B, C, Pl, Lat, Lon)
            - mask (torch.Tensor, optional): 掩码张量，形状为 (B, 1, Pl, Lat, Lon) 或 (B, Pl, Lat, Lon)
        输出 x (torch.Tensor): 输出张量，形状与输入 x 相同

    Example:
        >>> fuser = XiheLocalSIEFuser(
        ...     dim=192,
        ...     input_resolution=(13, 128, 256),
        ...     depth=2,
        ...     num_heads=6,
        ...     window_size=(1, 6, 12)
        ... )
        >>> from types import SimpleNamespace
        >>> B, C, Pl, Lat, Lon = 2, 192, 13, 128, 256
        >>> x = torch.randn(B, C, Pl, Lat, Lon)
        >>> mask = torch.ones(B, 1, Pl, Lat, Lon, dtype=torch.bool)
        >>> obj = SimpleNamespace(x=x, mask=mask)
        >>> out = fuser(obj)
        >>> out.shape
        torch.Size([2, 192, 13, 128, 256])
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth=2,
        num_heads=6,
        window_size=(1,6,12),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        # self.depth = depth

        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    dim=dim,
                    input_resolution=input_resolution, #3d windows
                    style="XihelocalTransformer",
                )
                for i in range(depth)
            ]
        )

    def forward(self, obj):
        # x=obj.x
        # mask=obj.mask

        if isinstance(obj, dict):
            # 字典方式访问
            x=obj["x"]
            mask = obj["mask"].clone().detach().float()
    
        # 判断是否为对象（非字典的其他类型）
        else:
            # 对象方式访问        
            x=obj.x
            mask=obj.mask
            obj={
                "x":x,
                "mask":mask,
            }   
        for blk in self.blocks:
            x = blk(x) if mask is None else blk(obj)
            # obj.x=x
            obj["x"]=x             
        return x
