import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange

from onescience.modules import OneEmbedding, OneFuser


class FourCastNet(nn.Module):
    """
    FourCastNet 的主模型实现。

    该模型使用以下组件完成输入编码与主干特征提取：

    - `OneEmbedding(style="FourCastNetEmbedding")`
      - 将二维气象场切分为二维 patch token 序列
    - `OneFuser(style="FourCastNetFuser")`
      - 在二维 patch 网格上重复执行 AFNO 频域混合与 MLP 通道混合

    在当前实现中：

    - 输入为二维单时刻气象场 `(Batch, Channels, Height, Width)`
    - patch embedding 输出会加上可学习位置编码
    - token 序列随后恢复成 `(PatchGridHeight, PatchGridWidth)` 二维网格
    - 多层 `FourCastNetFuser` 在 patch 网格上完成主干特征提取
    - 最终通过线性头恢复回目标变量场

    Args:
        img_size (tuple[int, int]):
            输入空间尺寸 `(Height, Width)`。
        patch_size (tuple[int, int]):
            patch 切分尺寸 `(PatchHeight, PatchWidth)`。
        in_chans (int):
            输入变量通道数。
        out_chans (int):
            输出变量通道数。
        embed_dim (int):
            patch embedding 特征维度。
        depth (int):
            主干 `FourCastNetFuser` 堆叠层数。
        mlp_ratio (float):
            每层 MLP 隐层放大倍数。
        drop_rate (float):
            dropout 比例。
        drop_path_rate (float):
            按层递增的 Stochastic Depth 最大比例。
        num_blocks (int):
            AFNO 的通道分块数。
        sparsity_threshold (float):
            AFNO 的 soft shrink 阈值。
        hard_thresholding_fraction (float):
            AFNO 保留的频率模式比例。
    """

    def __init__(
        self,
        img_size=(720, 1440),
        patch_size=(8, 8),
        in_chans=19,
        out_chans=19,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        drop_path = np.linspace(0, drop_path_rate, depth).tolist()

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_grid_height = img_size[0] // self.patch_size[0]
        self.patch_grid_width = img_size[1] // self.patch_size[1]

        self.patch_embed = OneEmbedding(
            style="FourCastNetEmbedding",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList([
            OneFuser(
                style="FourCastNetFuser",
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path[i],
                num_blocks=num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            for i in range(depth)
        ])

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                输入张量，形状为 `(Batch, Channels, Height, Width)`。

        Returns:
            torch.Tensor:
                输出张量，形状为 `(Batch, out_chans, Height, Width)`。
        """
        Batch = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(Batch, self.patch_grid_height, self.patch_grid_width, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.patch_grid_height,
            w=self.patch_grid_width,
        )
        return x
