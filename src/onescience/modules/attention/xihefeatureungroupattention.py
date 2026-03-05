import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp
class FeatureUngroupingAttention(nn.Module):
    """
    全局 SIE 的第三步：特征解组模块，将全局 group 特征融合回局部 patch tokens。

    Args:
        dim (int): 输入通道数 C。
        num_heads (int): 多头注意力的头数，默认为 12。
        qkv_bias (bool): 是否在 QKV 上添加可学习的偏置，默认为 True。
        attn_drop (float): 注意力权重的 dropout 比例，默认为 0.0。
        proj_drop (float): 输出投影的 dropout 比例，默认为 0.0。
        LN (nn.Module): 归一化层类型，默认为 nn.LayerNorm。
        drop_layer (nn.Module): dropout 层类型，默认为 nn.Dropout。

    形状:
        输入 obj: 包含以下属性的对象
            - y (torch.Tensor): 原始 patch tokens，形状为 (B, N, C)，其中 N = Pl × Lat × Lon
            - x (torch.Tensor): 经过 Group Propagation 的 group vectors，形状为 (B, G, C)
        输出 x_out (torch.Tensor): 融合全局信息后的 patch tokens，形状为 (B, N, C)

    Example:
        >>> ungrouping = FeatureUngroupingAttention(
        ...     dim=192,
        ...     num_heads=12
        ... )
        >>> from types import SimpleNamespace
        >>> B, N, C = 2, 425984, 192  # N = 13*128*256
        >>> G = 32  # group 数量
        >>> x_patch = torch.randn(B, N, C)  # patch tokens
        >>> G_tilde = torch.randn(B, G, C)  # group vectors
        >>> obj = SimpleNamespace(y=x_patch, x=G_tilde)
        >>> out = ungrouping(obj)
        >>> out.shape
        torch.Size([2, 425984, 192])
    """

    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        LN=nn.LayerNorm,
        drop_layer=nn.Dropout,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads     
     
        self.norm_x = LN(dim)  # 对 patch tokens 做归一化
        self.norm_g = LN(dim)  # 对 group vectors 做归一化
        # Cross-Attention (Q=patch tokens, K/V=group vectors)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop,batch_first=True
        )
        
        # 注意力输出的投影层
        self.attn_proj = nn.Linear(dim, dim)
         # 拼接后的融合层
        self.concat_proj = nn.Linear(2 * dim, dim)
        self.proj_drop = drop_layer(proj_drop)

    def forward(self, obj,mask=None):
        """
        x: (B, N, C)  patch tokens
        G_tilde: (B, G, C)  group vectors
        """
        x=obj.y
        G_tilde=obj.x
        
        B, N, C = x.shape
        _, G, _ = G_tilde.shape

        # 归一化
        x_norm = self.norm_x(x)
        G_norm = self.norm_g(G_tilde)

        x_out, _ = self.attn(query=x_norm, key=G_norm, value=G_norm)
        x_out = self.proj_drop(self.attn_proj(x_out))
        
        # 拼接 [U, x] 并线性映射回原维度 C
        x_concat = torch.cat([x_out, x], dim=-1)   # (B, N, 2C)   
        x_out = self.proj_drop(self.concat_proj(x_concat))  # (B, N, C)


        return x_out