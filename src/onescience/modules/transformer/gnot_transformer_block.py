import torch
import torch.nn as nn
import torch.nn.functional as F
from onescience.modules.mlp import StandardMLP as MLP
from onescience.modules.attention.linearattention import LinearAttention

class GNOTTransformerBlock(nn.Module):
    """
    GNOT Transformer 编码器块 (MoE 风格)。

    该模块结合了线性注意力机制 (Linear Attention) 和混合专家模型 (MoE) 的 MLP 结构。
    它不仅处理几何特征的自注意力 (Self-Attention)，还通过交叉注意力 (Cross-Attention) 融合物理场观测信息。

    主要组件：
    1. **Cross Attention**: 融合几何特征 (x) 和物理特征 (y)。
    2. **MoE MLP 1**: 第一层混合专家前馈网络，由门控网络 (GateNet) 控制专家的加权求和。
    3. **Self Attention**: 几何特征内部的线性自注意力。
    4. **MoE MLP 2**: 第二层混合专家前馈网络。

    Args:
        num_heads (int): 注意力头的数量。
        hidden_dim (int): 隐藏层特征维度。
        dropout (float): Dropout 概率。
        act (str, optional): 激活函数类型。默认值: "gelu"。
        mlp_ratio (int, optional): MLP 隐藏层维度相对于 hidden_dim 的倍数。默认值: 4。
        space_dim (int, optional): 空间坐标维度 (用于门控网络)。默认值: 2。
        n_experts (int, optional): MoE 中的专家数量。默认值: 3。

    形状:
        输入 x: (B, N, C) - 几何特征
        输入 y: (B, N, C) - 物理场/辅助特征
        输入 pos: (B, N, space_dim) - 空间坐标，用于计算 MoE 的门控分数
        输出: (B, N, C) - 更新后的几何特征

    Example:
        >>> block = GNOTTransformerBlock(
        ...     num_heads=4, hidden_dim=64, dropout=0.1, 
        ...     space_dim=2, n_experts=3
        ... )
        >>> x = torch.randn(2, 1024, 64)
        >>> y = torch.randn(2, 1024, 64)
        >>> pos = torch.randn(2, 1024, 2)
        >>> out = block(x, y, pos)
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
        space_dim=2,
        n_experts=3,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)

        # Linear Attention
        self.selfattn = LinearAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
        )
        self.crossattn = LinearAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
        )
        
        self.resid_drop1 = nn.Dropout(dropout)
        self.resid_drop2 = nn.Dropout(dropout)

        ## MLP in MOE
        self.n_experts = n_experts
        
        self.moe_mlp1 = nn.ModuleList([
            MLP(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim * mlp_ratio],
                activation=act,
                use_bias=True
            )
            for _ in range(self.n_experts)
        ])

        self.moe_mlp2 = nn.ModuleList([
            MLP(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim * mlp_ratio],
                activation=act,
                use_bias=True
            )
            for _ in range(self.n_experts)
        ])
        
        self.gatenet = MLP(
            input_dim=space_dim,
            output_dim=self.n_experts,
            hidden_dims=[hidden_dim * mlp_ratio, hidden_dim * mlp_ratio],
            activation=act,
            use_bias=True
        )

    def forward(self, x, y, pos):
        ## point-wise gate for moe
        gate_score = F.softmax(self.gatenet(pos), dim=-1).unsqueeze(2)
        
        ## cross attention between geo and physics observation
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln2(y)))
        
        ## moe mlp 1
        x_moe1 = torch.stack(
            [self.moe_mlp1[i](x) for i in range(self.n_experts)], dim=-1
        )
        x_moe1 = (gate_score * x_moe1).sum(dim=-1, keepdim=False)
        x = x + self.ln3(x_moe1)
        
        ## self attention among geo
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        
        ## moe mlp 2
        x_moe2 = torch.stack(
            [self.moe_mlp2[i](x) for i in range(self.n_experts)], dim=-1
        )
        x_moe2 = (gate_score * x_moe2).sum(dim=-1, keepdim=False)
        x = x + self.ln5(x_moe2)
        return x