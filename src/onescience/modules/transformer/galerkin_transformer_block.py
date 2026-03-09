import torch
import torch.nn as nn
from typing import Optional

from onescience.modules import OneMlp, OneAttention

class Galerkin_Transformer_block(nn.Module):
    """
    Galerkin Transformer 编码器块。

    该模块是 Galerkin Transformer 的核心组件。它使用基于 Galerkin 投影的线性注意力机制 
    (Linear Attention) 来代替标准点积注意力，从而将 Transformer 处理序列的时间和空间
    复杂度从 O(N^2) 降低为 O(N)，使其能够高效处理大规模的网格和物理场数据。

    该块包含以下主要组件：
    1. **LayerNorm**: 在注意力层和 MLP 层之前应用 (Pre-Norm 结构)。
    2. **Linear Attention**: 使用 Galerkin 类型的线性注意力机制。注意，在该实现中 Attention 
       接收两个输入 (`ln_1(fx)` 和 `ln_1a(fx)`)，以分别处理 Query/Key 与 Value 的归一化特征。
    3. **Feed-Forward Network (MLP)**: 标准的多层感知机，用于特征的非线性逐点映射。
    4. **Residual Connections**: 在 Attention 和 MLP 后均应用残差连接。

    Args:
        num_heads (int): 注意力头的数量。
        hidden_dim (int): 隐藏层特征维度 (Embedding Dimension)。
        dropout (float): Dropout 概率。
        act (str, optional): 激活函数类型。默认值: "gelu"。
        mlp_ratio (int, optional): MLP 隐藏层维度相对于 hidden_dim 的扩展倍数。默认值: 4。
        last_layer (bool, optional): 是否为网络的最后一层。如果是，将应用额外的 LayerNorm 和线性投影。默认值: False。
        out_dim (int, optional): 如果是最后一层，输出的特征维度。默认值: 1。

    形状:
        输入 fx: (B, N, C) 或 (B, C, H, W) 取决于 LinearAttention 的具体实现和输入数据的维度排列。
        输出: 通常与输入形状一致，除非是最后一层且 out_dim 改变了通道数。

    Example:
        >>> block = Galerkin_Transformer_block(
        ...     num_heads=4,
        ...     hidden_dim=64,
        ...     dropout=0.1,
        ...     mlp_ratio=4
        ... )
        >>> x = torch.randn(2, 1024, 64)
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([2, 1024, 64])
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
    ):
        super().__init__()
        self.last_layer = last_layer
        
        # Norm layers
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_1a = nn.LayerNorm(hidden_dim)
        
        # Attention
        self.Attn = OneAttention(
            style="LinearAttention",
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            attn_type="galerkin",
        )
        
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        self.mlp = OneMlp(
            style="StandardMLP",
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim * mlp_ratio],
            activation=act,
            use_bias=True, 
        )
        
        # Output Projection 
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        # Attention Residual: x + Attn(LN(x), LN(x))
        fx = self.Attn(self.ln_1(fx), self.ln_1a(fx)) + fx
        
        # MLP Residual: x + MLP(LN(x))
        fx = self.mlp(self.ln_2(fx)) + fx
        
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx