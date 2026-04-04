import torch.nn as nn
from timm.models.layers import DropPath
from onescience.modules.fc.onefc import OneFC
from onescience.modules.afno.oneafno import OneAFNO


class FourCastNetFuser(nn.Module):
    """
        FourCastNet 的主干特征融合模块。

        该模块接收二维 patch token 网格 `(Height, Width)` 上的特征，
        使用 AFNO 完成全局频域混合，再使用逐位置 MLP 完成通道混合。

        结构顺序为：

        - `LayerNorm`
        - `OneAFNO(style="FourCastNetAFNO2D")`
        - 可选中间残差连接
        - `LayerNorm`
        - `OneFC(style="FourCastNetFC")`
        - DropPath 与最终残差连接

        与 PanguFuser 不同，该模块处理的不是展平 token 序列，而是已经恢复为
        二维 patch 网格的特征张量。

        Args:
            dim (int):
                输入与输出特征维度。
            mlp_ratio (float):
                MLP 隐层相对于 `dim` 的放大倍数。
            drop (float):
                MLP dropout 比例。
            drop_path (float):
                Stochastic Depth 比例。
            act_layer (nn.Module):
                MLP 激活函数类型。
            norm_layer (nn.Module):
                归一化层类型。
            double_skip (bool):
                是否启用中间残差连接。
            num_blocks (int):
                AFNO 的通道分块数。
            sparsity_threshold (float):
                AFNO 中的 soft shrink 阈值。
            hard_thresholding_fraction (float):
                AFNO 中保留的频率模式比例。

        形状:
            输入:
                `x` 形状为 `(Batch, Height, Width, dim)`
            输出:
                `x` 形状为 `(Batch, Height, Width, dim)`，与输入相同

        Examples:
            >>> Batch = 2
            >>> Height = 90
            >>> Width = 180
            >>> dim = 768
            >>> block = FourCastNetFuser(
            ...     dim=dim,
            ...     mlp_ratio=4.0,
            ...     double_skip=True,
            ...     num_blocks=8,
            ...     sparsity_threshold=0.01,
            ...     hard_thresholding_fraction=1.0,
            ... )
            >>> x = torch.randn(Batch, Height, Width, dim)
            >>> out = block(x)
            >>> out.shape
            torch.Size([2, 90, 180, 768])
    """

    def __init__(
        self,
        dim=768,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = OneAFNO(
            style="FourCastNetAFNO2D",
            hidden_size=dim,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = OneFC(
            style="FourCastNetFC",
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x):
        Residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + Residual
            Residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + Residual
        return x
