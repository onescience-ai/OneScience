from collections.abc import Sequence

import torch
from torch import nn

from onescience.modules.embedding.oneembedding import OneEmbedding
from onescience.modules.sample.onesample import OneSample
from onescience.modules.recovery.onerecovery import OneRecovery
from onescience.modules.transformer.onetransformer import OneTransformer

class FengWuDecoder(nn.Module):
    """
    FengWu 模型的解码器模块。
    
    采用"中间层处理 → 上采样 → 精细层处理 → 跳跃连接融合 → Patch 恢复"的流水线结构。
    接收编码器输出的中间分辨率特征与跳跃连接特征，经过多尺度 Transformer 处理后，
    输出最终气象预报场。
    
    Args:
        output_resolution (tuple[int, int], optional): 输出特征图分辨率 (lat, lon)，
            默认为 (181, 360)。
        middle_resolution (tuple[int, int], optional): 中间层特征图分辨率 (lat, lon)，
            默认为 (91, 180)，约为 output_resolution 的 1/2。
        out_chans (int, optional): 最终输出的气象变量通道数，默认为 37。
        img_size (tuple[int, int], optional): 原始输入图像分辨率 (lat, lon)，
            用于 PatchRecovery 还原，默认为 (721, 1440)。
        patch_size (tuple[int, int], optional): Patch 大小，用于 PatchRecovery，
            默认为 (4, 4)。
        dim (int, optional): 基础嵌入维度，中间层使用 dim*2，默认为 192。
        depth (int, optional): 输出分辨率处 Transformer Block 的层数，默认为 2。
        depth_middle (int, optional): 中间分辨率处 Transformer Block 的层数，默认为 6。
        num_heads (tuple[int, int] 或 int, optional): 各阶段注意力头数 (中间层, 输出层)，
            默认为 (6, 12)。若为单个 int，则两阶段共用。
        window_size (tuple[int, int], optional): 窗口注意力的窗口大小 (Wlat, Wlon)，
            默认为 (6, 12)。
        mlp_ratio (float, optional): MLP 隐层相对于嵌入维度的扩展倍数，默认为 4.0。
        qkv_bias (bool, optional): 是否为 QKV 投影添加偏置项，默认为 True。
        qk_scale (float, optional): QK 点积的缩放系数，默认为 None，
            自动使用 head_dim ** -0.5。
        drop (float, optional): MLP 的 Dropout 比例，默认为 0.0。
        attn_drop (float, optional): 注意力权重的 Dropout 比例，默认为 0.0。
        drop_path (float 或 Sequence[float], optional): 各 Block 的 DropPath 比例，
            若为 Sequence，前 depth 个分配给输出层，剩余分配给中间层，默认为 0.0。
        norm_layer (nn.Module, optional): 归一化层类型，默认为 nn.LayerNorm。
    
    形状:
        - 输入 inp[0] (x):    (B, middle_lat * middle_lon, dim * 2)
        - 输入 inp[1] (skip): (B, output_lat, output_lon, dim)
        - 输出:               (B, out_chans, img_lat, img_lon)
    
    Examples:
        >>> # 典型 FengWu 解码器配置
        >>> # middle_resolution=(91, 180)，output_resolution=(181, 360)
        >>> # middle_lat * middle_lon = 91 * 180 = 16380
        >>> # output_lat * output_lon = 181 * 360 = 65160
        >>> decoder = FengWuDecoder(
        ...     output_resolution=(181, 360),
        ...     middle_resolution=(91, 180),
        ...     out_chans=37,
        ...     img_size=(721, 1440),
        ...     patch_size=(4, 4),
        ...     dim=192,
        ...     depth=2,
        ...     depth_middle=6,
        ...     num_heads=(6, 12),
        ...     window_size=(6, 12),
        ... )
        >>> B = 2
        >>> x    = torch.randn(B, 16380, 384)   # (B, middle_lat*middle_lon, dim*2)
        >>> skip = torch.randn(B, 181, 360, 192) # (B, output_lat, output_lon, dim)
        >>> out = decoder([x, skip])
        >>> out.shape
        torch.Size([2, 37, 721, 1440])
    """
    def __init__(
        self,
        output_resolution=(181, 360),
        middle_resolution=(91,180),
        out_chans=37,
        img_size=(721, 1440),
        patch_size=(4, 4),
        dim=192,
        depth=2,
        depth_middle=6,
        num_heads=(6, 12),
        window_size=(6, 12),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.out_chans = out_chans
        self.dim = dim
        self.output_resolution = output_resolution
        self.depth = depth
        self.depth_middle = depth_middle
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path
        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads

        self.blocks_middle = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer2DBlock",
                    dim=dim * 2,
                    input_resolution=middle_resolution,
                    num_heads=num_heads_middle,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path_middle[i] if isinstance(drop_path_middle, Sequence) else drop_path_middle,
                )
                for i in range(depth_middle)
            ]
        )

        self.upsample = OneSample(
            style="PanguUpSample2D",
            in_dim=dim * 2,
            out_dim=dim,
            input_resolution=middle_resolution,
            output_resolution=output_resolution
        )

        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer2DBlock",
                    dim=dim,
                    input_resolution=output_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,

                )
                for i in range(depth)
            ]
        )

        self.patchrecovery2d = OneRecovery(
            style="pangupatchrecovery2d",
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=2 * dim, 
            out_chans=out_chans
        )


    def forward(self, inp):
        x, skip = inp[0], inp[1]
        B, Lat, Lon, C = skip.shape
        for blk in self.blocks_middle:
            x = blk(x)
        x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        output = torch.concat([x, skip.reshape(B, -1, C)], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Lat, Lon)
        output = self.patchrecovery2d(output)
        return output