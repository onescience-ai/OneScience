import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from typing import Sequence

from onescience.modules.func_utils.fuxi_utils import get_pad2d


class FuxiEmbedding(nn.Module):
    """
    FuXi 模型的三维 Patch Embedding 模块。
    
    使用 3D 卷积将 (时间步, 纬度, 经度) 三维气象场划分为不重叠的 Patch 并投影到
    嵌入空间，是 FuXi 模型编码器的入口层。与 Pangu-Weather 将气压层和地表变量
    分开处理不同，FuXi 将多帧气象场沿时间轴堆叠后统一做三维 Patch 划分。
    
    Args:
        img_size (tuple[int, int, int], optional): 输入数据的空间尺寸 (T, lat, lon)，
            其中 T 为时间步数（通常为 2，对应当前时刻与前一时刻），默认为 (2, 721, 1440)。
        patch_size (tuple[int, int, int], optional): 3D Patch 大小 (Pt, Plat, Plon)，
            默认为 (2, 4, 4)，时间维度通常设为与 T 相同以合并时间步。
        in_chans (int, optional): 输入气象变量通道数，默认为 70。
        embed_dim (int, optional): Patch 嵌入维度，默认为 1536。
        norm_layer (nn.Module 或 None, optional): 嵌入后的归一化层类型，
            为 None 时跳过归一化，默认为 nn.LayerNorm。
        **kwargs: 额外参数（忽略，兼容统一接口）。
    
    形状:
        - 输入 x:  (B, C, T, lat, lon)
            其中 C = in_chans，T = img_size[0]
        - 输出:    (B, embed_dim, T//Pt, lat//Plat, lon//Plon)
            即 (B, embed_dim, nT, nLat, nLon)
    
    Examples:
        >>> # 典型 FuXi 配置：2帧输入，70个气象变量
        >>> # patch_size=(2,4,4)，时间维度完全合并
        >>> # nT   = 2 // 2 = 1
        >>> # nLat = 721 // 4 = 180
        >>> # nLon = 1440 // 4 = 360
        >>> embedding = FuxiEmbedding(
        ...     img_size=(2, 721, 1440),
        ...     patch_size=(2, 4, 4),
        ...     in_chans=70,
        ...     embed_dim=1536,
        ... )
        >>> x = torch.randn(2, 70, 2, 721, 1440)  # (B, C, T, lat, lon)
        >>> out = embedding(x)
        >>> out.shape
        torch.Size([2, 1536, 1, 180, 360])
    """
    def __init__(self, 
                 img_size=(2, 721, 1440), 
                 patch_size=(2, 4, 4), 
                 in_chans=70, 
                 embed_dim=1536, 
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        assert T == self.img_size[0] and Lat == self.img_size[1] and Lon == self.img_size[2], \
            f"Input image size ({T}*{Lat}*{Lon}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x

