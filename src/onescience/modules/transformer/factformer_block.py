import torch
import torch.nn as nn
from typing import Optional, List, Union

# 替换为标准 MLP 模块
from onescience.modules.mlp import StandardMLP as MLP
from onescience.modules.attention.factattention import FactAttention2D, FactAttention3D
from onescience.modules.attention.multiheadattention import MultiHeadAttention as Attention

FACT_ATTENTION = {
    "structured_1D": Attention,
    "structured_2D": FactAttention2D,
    "structured_3D": FactAttention3D,
}

class Factformer_block(nn.Module):
    """
    FactFormer 编码器块 (FactFormer Encoder Block)。

    这是 FactFormer 模型的核心组件，采用标准的 Pre-Norm Transformer 结构。
    它包含两个主要子层：
    1. **多头注意力层 (Multi-Head Attention)**: 根据 `geotype` 参数选择使用标准注意力 (1D) 或分解注意力 (FactAttention 2D/3D)。
    2. **前馈网络层 (Feed-Forward Network)**: 使用 MLP 对特征进行逐点变换。

    每个子层周围都应用了残差连接 (Residual Connection) 和层归一化 (LayerNorm)。
    如果是最后一层，还可以选择附加一个线性投影层输出最终维度。

    Args:
        num_heads (int): 注意力头的数量。
        hidden_dim (int): 隐藏层特征维度 (Embedding Dimension)。
        dropout (float): Dropout 概率。
        act (str, optional): 激活函数类型。默认值: "gelu"。
        mlp_ratio (int, optional): MLP 隐藏层维度相对于 hidden_dim 的倍数。默认值: 4。
        last_layer (bool, optional): 是否为网络的最后一层。如果是，将应用额外的 LayerNorm 和线性投影层。默认值: False。
        out_dim (int, optional): 如果是最后一层，输出的特征维度。默认值: 1。
        geotype (str, optional): 几何类型，用于选择 Attention 实现。可选 ["structured_1D", "structured_2D", "structured_3D"]。默认值: "unstructured" (但在 FACT_ATTENTION 字典中未定义，需注意)。
        shapelist (list[int], optional): 输入网格的形状列表 (H, W) 或 (D, H, W)，用于 FactAttention。

    形状:
        输入 fx:
            - 如果是 1D/Unstructured: (B, N, C)
            - 如果是 2D: (B, C, H, W) 或 (B, N, C) (取决于具体的 Attention 实现，通常 FactAttention 需要特定的维度排列)
        输出: 
            - 形状通常与输入保持一致 (除非 last_layer=True 且 out_dim 改变了维度)。

    Example:
        >>> # 2D 结构化网格示例
        >>> block = Factformer_block(
        ...     num_heads=4,
        ...     hidden_dim=64,
        ...     dropout=0.1,
        ...     geotype="structured_2D",
        ...     shapelist=[32, 32]
        ... )
        >>> # 假设 FactAttention2D 期望输入 (B, C, H, W) 或 (B, H*W, C)
        >>> # 这里构造 dummy 输入
        >>> x = torch.randn(2, 32*32, 64) 
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([2, 1024, 64])
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        geotype: str = "structured_2D", 
        shapelist: Optional[List[int]] = None,
    ):
        super().__init__()
        self.last_layer = last_layer
        
        # 1. Pre-Norm
        self.ln_1 = nn.LayerNorm(hidden_dim)

        # 2. Attention Mechanism
        if geotype not in FACT_ATTENTION:

             AttnClass = Attention
        else:
             AttnClass = FACT_ATTENTION[geotype]

        self.Attn = AttnClass(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            shapelist=shapelist,
        )
        
        # 3. MLP Block (Pre-Norm)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        self.mlp = MLP(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim * mlp_ratio],
            activation=act,
            use_bias=True, 
        )
        
        # 4. Optional Last Layer Projection
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        # Attention Residual
        fx = self.Attn(self.ln_1(fx)) + fx
        
        # MLP Residual
        fx = self.mlp(self.ln_2(fx)) + fx
        
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx