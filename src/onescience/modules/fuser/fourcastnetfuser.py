import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from onescience.modules.fc.onefc import OneFC
from onescience.modules.afno.oneafno import OneAFNO

class FourCastNetFuser(nn.Module):
    """
        FourCastNet 的核心 Transformer Block。

        以 AFNO 替代自注意力机制的 Transformer Block，结构为
        "AFNO 频域混合 → MLP 通道混合"，并支持双残差连接（double skip）。
        开启 double_skip 时，AFNO 输出与输入残差相加后再送入 MLP，
        相当于在标准残差连接基础上额外引入一次中间残差，有助于深层网络的梯度传播。

        Args:
            dim (int, optional): 输入 token 的通道数（嵌入维度），默认为 768。
            mlp_ratio (float, optional): MLP 隐层相对于 dim 的扩展倍数，默认为 4.0。
            drop (float, optional): MLP 的 Dropout 比例，默认为 0.0。
            drop_path (float, optional): Stochastic Depth 的比例，默认为 0.0。
            act_layer (nn.Module, optional): MLP 的激活函数，默认为 nn.GELU。
            norm_layer (nn.Module, optional): 归一化层类型，默认为 nn.LayerNorm。
            double_skip (bool, optional): 是否启用双残差连接，默认为 True。
                True:  x → AFNO → (x + residual1) → MLP → (x + residual2)
                False: x → AFNO → MLP → (x + residual)
            num_blocks (int, optional): 传递给 AFNO 的通道分块数，默认为 8。
            sparsity_threshold (float, optional): 传递给 AFNO 的软阈值，默认为 0.01。
            hard_thresholding_fraction (float, optional): 传递给 AFNO 的频率保留比例，
                默认为 1.0。

        形状:
            - 输入 x: (B, H, W, C)，其中 C = dim
            - 输出:   (B, H, W, C)，形状与输入完全一致

        Examples:
            >>> # 典型 FourCastNet Block 配置，启用双残差连接
            >>> block = FourCastNetFuser(
            ...     dim=768,
            ...     mlp_ratio=4.0,
            ...     double_skip=True,
            ...     num_blocks=8,
            ...     sparsity_threshold=0.01,
            ...     hard_thresholding_fraction=1.0,
            ... )
            >>> x = torch.randn(2, 720, 1440, 768)  # (B, H, W, C)
            >>> out = block(x)
            >>> out.shape
            torch.Size([2, 720, 1440, 768])
    """
    def __init__(
            self,
            dim=768,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = OneAFNO(style="FourCastNetAFNO2D")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = OneFC(style="FourCastNetFC")
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x
