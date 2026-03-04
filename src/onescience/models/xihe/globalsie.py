import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp

#GLOBAL 1
class FeatureGrouping(nn.Module):
    """
    Global SIE -Step 1: Feature Grouping
    功能: 将高分辨率的局部 token 表示压缩为少量的 group 表示，用于构建全局上下文并缓解计算量；同时借助 mask_tokens 只让有效区域（如海洋）参与全局聚合，避免无效区域（如陆地）干扰

    Args:
        dim (int): 输入特征通道数.
        num_groups (int): group vectors数量.
        num_heads (int): 多头注意力的head数.
        qkv_bias (bool): 是否使用QKV bias.
        attn_drop (float): 注意力dropout.
        proj_drop (float): 输出投影dropout.

    形状:
        输入:
            x: (B, N, C),局部特征序列(N = H x W)
            mask_tokens: (B, N)，有效 token 掩码(1=有效,0=忽略）
        输出:
            G_prime: (B, G, C)，聚合后的 group 特征

    Returns:
        Tensor: 更新后的 group vectors,形状为 (B, G, C)。
    """

    def __init__(
        self,
        dim, 
        num_groups=32, 
        num_heads=8, 
        qkv_bias=True,
        attn_drop=0.0, 
        proj_drop=0.0,
        LN=nn.LayerNorm,
        drop_layer=nn.Dropout,
        ):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups  
        self.num_heads = num_heads  #自定义，分组多就可表示的更精细

        # 初始化 learnable group vectors (相当于 G_l)
        self.group_vectors = nn.Parameter(torch.randn(1, num_groups, dim))
        self.norm = LN(dim)
        # 多头注意力 (标准 vanilla Transformer Attention)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, bias=qkv_bias, batch_first=True
        )
        self.attn_drop = drop_layer(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = drop_layer(proj_drop)
    def forward(self, x,mask_tokens=None):
        """
        x: (B, N, C)  -> 来自 Local SIE 的特征
        """
        B, N, C = x.shape
        x = self.norm(x)  # (B, N, C)
        
        #  expand group vectors (batch 内共享同一份 group 参数)
        G = self.group_vectors.expand(B, -1, -1)  # (B, G, C)
        # Multi-Head Cross-Attention
        if mask_tokens is None:
            return None
        if mask_tokens.dim() == 4:              # (B,1,H,W)
            mask_tokens = mask_tokens.squeeze(1)
        if mask_tokens.dim() == 3:              # (B,H,W)
            mask_tokens = mask_tokens.reshape(B, -1)
        assert mask_tokens.shape == (B, N)
        # key_padding_mask = (mask_tokens == 0)   # True=忽略
        key_padding_mask = None if mask_tokens is None else (mask_tokens == 0)
        G_prime, _ = self.attn(query=G, key=x, value=x,key_padding_mask=key_padding_mask)
        #  输出更新后的 group vectors
        G_prime = self.proj_drop(self.proj(G_prime))  # (B, G, C)

        return G_prime

#GLOBAL 2   
class GroupPropagation(nn.Module):
    """
    Global SIE - Step 2: Group Propagation
    目的: 在 group 空间内进行信息传播与融合，使各个 group 表示获得更强的全局一致性与互补性，为后续将全局信息回灌到 patch tokens（Ungrouping）提供更完整的全局上下文.

    Args:
        dim (int): 输入特征通道数 C.
        num_groups (int): group vectors 数量 G.
        mlp_ratio (float): MLP 隐层扩展比例.
        drop (float): MLP dropout.
        act_layer (nn.Module): 激活函数层类型（默认 GELU).
        LN (nn.Module): 归一化层类型（默认 LayerNorm).

    形状:
        输入:
            x: (B, G, C)，输入 group vectors.
        输出:
            G_tilde: (B, G, C)，传播并融合后的 group vectors.

    Returns:
        Tensor: 输出 group vectors,形状为 (B, G, C).
    """

    def __init__(
        self,
        dim, 
        num_groups,
        mlp_ratio=4.0,
        drop=0.0, 
        act_layer=nn.GELU,
        LN=nn.LayerNorm
        ):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups

        # LayerNorm
        self.norm1 = LN(dim)
        self.norm2 = LN(dim)
        
        # Token-mixing MLP (在 group 维度上传播信息)
        mlp_token_dim = int(num_groups * mlp_ratio)
        self.mlp_token = Mlp(
            in_features=num_groups,
            hidden_features=mlp_token_dim,
            act_layer=act_layer,
            drop=drop,
        )        
    
       # Channel-mixing MLP (在 embedding 维度融合特征)
        mlp_channel_dim = int(dim * mlp_ratio)
        self.mlp_channel =Mlp(
            in_features=dim,
            hidden_features=mlp_channel_dim,
            act_layer=act_layer,
            drop=drop,
        )
        
    def forward(self, x):
        """
        x: (B, G, C) -> 输入 group vectors
        """
        B, G, C = x.shape
        shortcut=x
        

        # Step 1: Token mixing (group 维度传播信息)
        x = self.norm1(x)          # (B, G, C)
        x = x.transpose(1, 2)            # (B, C, G)
        x = self.mlp_token(x)            # (B, C, G) 先对group进行mlp
        x = x.transpose(1, 2)            # (B, G, C)
        x = shortcut + x              # 残差连接
        # Step 2: Channel mixing (embedding 维度融合)  
        y = self.norm2(x)            # (B, G, C)
        y = self.mlp_channel(y)          # (B, G, C) 在对channel进行mlp
        y = x + y              # 残差连接
        return y

#GLOBAL 3
class FeatureUngrouping(nn.Module):
    """
    Global SIE - Step 3: Feature Ungrouping
    目的: 将经过全局建模的 group 融合回高分辨率 patch tokens 中，使局部特征获得全局上下文约束，同时保持原有空间分辨率不变.

    Args:
        dim (int): 输入特征通道数.
        num_heads (int): 多头注意力的head数.
        qkv_bias (bool): 是否使用 QKV bias.
        attn_drop (float): 注意力 dropout.
        proj_drop (float): 输出投影 dropout.
        LN (nn.Module): 归一化层类型（默认 LayerNorm).
        drop_layer (nn.Module): dropout 层类型（默认 Dropout).

    形状:
        输入:
            x: (B, N, C),patch tokens(N = H x W)
            G_tilde: (B, G, C),group vectors
            mask: (可选) 未使用/预留
        输出:
            x_out: (B, N, C)，融合全局信息后的 patch tokens

    Returns:
        Tensor: 输出 patch tokens,形状为 (B, N, C)。
    """

    def __init__(
        self,
        dim,
        num_heads=8,
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
        
        # self.attn_drop = drop_layer(attn_drop)
        # self.proj_drop = drop_layer(proj_drop)

    def forward(self, x, G_tilde,mask=None):
        """
        x: (B, N, C)  patch tokens
        G_tilde: (B, G, C)  group vectors
        """
        B, N, C = x.shape
        _, G, _ = G_tilde.shape

        # 归一化
        x_norm = self.norm_x(x)
        G_norm = self.norm_g(G_tilde)

        # Cross-Attention: Q = x, K/V = groups
        x_out, _ = self.attn(query=x_norm, key=G_norm, value=G_norm)
        x_out = self.proj_drop(self.attn_proj(x_out))
        
        # 拼接 [U, x] 并线性映射回原维度 C
        x_concat = torch.cat([x_out, x], dim=-1)   # (B, N, 2C)
        
        # print("x_concat:",x_concat.shape,B,N,C)
        
        # x_out = self.concat_proj(x_concat)    
        x_out = self.proj_drop(self.concat_proj(x_concat))  # (B, N, C)

        # 残差连接：原始 patch + 全局信息
        # x_out = x + x_out

        return x_out

class GlobalSIE(nn.Module):
    """
    Global SIE (Global Spatial Information Exchange)
    目的: 通过“分组 → 传播 → 回灌”的三阶段结构，在保持原始token空间分辨率不变的前提下，引入全局范围的信息交互，用于建模远距离依赖与大尺度一致性特征。

    组成:
        Step 1 - Feature Grouping:
            将高分辨率 patch tokens 压缩为少量 group 表示，构建全局摘要.
        Step 2 - Group Propagation:
            在 group 空间内进行信息传播与融合，强化全局上下文.
        Step 3 - Feature Ungrouping:
            将全局信息回灌至 patch tokens,更新局部表示.

    Args:
        dim (int): 输入特征通道数.
        num_heads (int): 注意力 head 数（与 Local SIE 对齐）.
        qkv_bias (bool): 是否使用 QKV bias.
        num_groups (int): group vectors 数量.
        norm_layer (nn.Module): 归一化层类型（默认 LayerNorm).

    形状:
        输入:
            x: (B, N, C),token 序列(N = H x W)
            mask: (可选) (B, N)，有效区域掩码（如 ocean-land mask)
        输出:
            x: (B, N, C)，融合全局信息后的 token 序列

    Returns:
        Tensor: 输出特征，形状为 (B, N, C)。
    """
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        num_groups=32,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim=dim
        self.num_heads=num_heads
        self.num_group=num_groups

        
        self.feature_grouping = FeatureGrouping(
            dim=dim,
            num_groups=num_groups,       # 超参数，可调整，靠输入数据决定
            num_heads=num_heads,         # 和 local 对齐
            qkv_bias=qkv_bias,
            
        )
        self.group_propagation = GroupPropagation(
            dim=dim,
            num_groups=num_groups,                
            mlp_ratio=4.0,               # 和 local 对齐
            drop=0.0,                    # 和 local 对齐
            act_layer=nn.GELU           
        ) 
        self.feature_ungrouping = FeatureUngrouping(        
            dim=dim,
            num_heads=num_heads,         # 和 local 对齐
            qkv_bias=qkv_bias           # 和 local 对齐            
        )

    def forward(self, x,mask=None):
        y=x
        x=self.feature_grouping(x,mask_tokens=mask)
        x=self.group_propagation(x)
        x=self.feature_ungrouping(y,x)
 
        return x
    