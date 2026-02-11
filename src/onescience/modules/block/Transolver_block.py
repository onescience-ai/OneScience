import torch
import torch.nn as nn
import numpy as np
from timm.layers import trunc_normal_
from onescience.modules.mlp import StandardMLP as MLP
from onescience.modules.attention.Physics_Attention import Physics_Attention_Irregular_Mesh
from onescience.modules.attention.Physics_Attention import Physics_Attention_Irregular_Mesh_plus
from onescience.modules.attention.Physics_Attention import (
    Physics_Attention_Structured_Mesh_1D,
)
from onescience.modules.attention.Physics_Attention import (
    Physics_Attention_Structured_Mesh_2D,
)
from onescience.modules.attention.Physics_Attention import (
    Physics_Attention_Structured_Mesh_3D,
)

PHYSICS_ATTENTION = {
    "unstructured": Physics_Attention_Irregular_Mesh,
    "unstructured_plus": Physics_Attention_Irregular_Mesh_plus,
    "structured_1D": Physics_Attention_Structured_Mesh_1D,
    "structured_2D": Physics_Attention_Structured_Mesh_2D,
    "structured_3D": Physics_Attention_Structured_Mesh_3D,
}

class Transolver_block(nn.Module):
    """
    Transolver 编码器块 (Transolver Encoder Block)。

    

    这是 Transolver 模型的核心构建单元，采用标准的 Transformer Encoder 架构，但集成了物理感知的注意力机制（Physics Attention）。
    每个块包含两个主要子层：
    1.  **物理注意力层**: 根据几何类型（结构化或非结构化网格）选择相应的注意力机制，用于捕捉物理场中的空间依赖关系。
    2.  **前馈网络 (MLP)**: 使用标准的多层感知机进行特征变换和非线性映射。

    两个子层都采用了 Pre-LayerNorm 结构（即先归一化再进入层）和残差连接。

    Args:
        num_heads (int): 注意力头的数量。
        hidden_dim (int): 隐藏层的特征维度（输入和输出维度）。
        dropout (float): Dropout 概率，用于防止过拟合。
        act (str, optional): 激活函数类型，例如 'gelu', 'relu'。默认值: 'gelu'。
        mlp_ratio (float, optional): MLP 隐藏层维度相对于 hidden_dim 的倍率。默认值: 4。
        last_layer (bool, optional): 是否为最后一个 Block。如果是，会额外包含一个 LayerNorm 和线性投影层用于输出。默认值: False。
        out_dim (int, optional): 如果是 last_layer，则指定最终的输出维度。默认值: 1。
        slice_num (int, optional): 用于物理注意力机制的分片数量（针对特定几何类型）。默认值: 32。
        geotype (str, optional): 几何类型，决定使用的 Attention 变体。
            支持: 'unstructured', 'unstructured_plus', 'structured_1D', 'structured_2D', 'structured_3D'。默认值: 'unstructured'。
        shapelist (list, optional): 网格形状列表，用于结构化网格的注意力计算。默认值: None。

    形状:
        输入 fx: (B, N, C)，其中 B 是批次大小，N 是节点/网格点数量，C 是 hidden_dim。
        输出:
            - 如果 last_layer=False: (B, N, C)，形状与输入相同。
            - 如果 last_layer=True: (B, N, out_dim)，经过投影后的输出。

    Example:
        >>> # 初始化一个非结构化网格的 Transolver Block
        >>> block = Transolver_block(
        ...     num_heads=8,
        ...     hidden_dim=128,
        ...     dropout=0.1,
        ...     geotype='unstructured'
        ... )
        >>> x = torch.randn(2, 1024, 128) # [Batch, Nodes, Dim]
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 1024, 128])
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
        slice_num=32,
        geotype="unstructured",
        shapelist=None,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        # 
        self.Attn = PHYSICS_ATTENTION[geotype](
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            shapelist=shapelist,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.mlp = MLP(
            input_dim=hidden_dim,
            hidden_dims=[int(hidden_dim * mlp_ratio)], # 中间膨胀层
            output_dim=hidden_dim,
            activation=act,
            dropout_rate=dropout, # 传入 Dropout 配置
            use_bias=True
        )

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        # Attention Sub-layer with Residual
        fx = self.Attn(self.ln_1(fx)) + fx
        
        # MLP Sub-layer with Residual
        fx = self.mlp(self.ln_2(fx)) + fx

        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx