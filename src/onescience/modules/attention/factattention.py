import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# ==========================================
# 辅助组件 (Internal Utilities)
# ==========================================

class _PoolingReducer(nn.Module):
    """
    内部组件：用于降维和特征压缩的池化层。
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.to_in(x)
        ndim = len(x.shape)
        if ndim > 3:
            x = x.mean(dim=tuple(range(2, ndim - 1)))
        x = self.out_ffn(x)
        return x 


class _FactAttnWeight(nn.Module):
    """
    内部组件：因子化注意力权重计算器。
    """
    def __init__(self, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        q = self.to_q(x)
        k = self.to_k(x)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        return attn 


# ==========================================
# 因子化注意力 (Factorized Attention)
# ==========================================

class FactAttention2D(nn.Module):
    """
    2D 因子化注意力 (Factorized Attention 2D)。

    

    专为 2D 网格结构数据设计的高效注意力机制。
    它不直接在展开的序列上计算 O((HW)^2) 的注意力，而是将特征分解为 X 轴和 Y 轴特征，
    分别计算注意力权重，然后通过爱因斯坦求和 (einsum) 进行特征聚合。

    **注意**：由于内部实现涉及 Reshape 操作，输入维度 `dim` 必须等于 `heads * dim_head`。

    Args:
        dim (int): 输入特征维度。
        heads (int, optional): 注意力头数。默认值: 8。
        dim_head (int, optional): 每个头的维度。默认值: 64。
        dropout (float, optional): Dropout 概率。默认值: 0.0。
        shapelist (tuple or list): 输入数据的网格形状 (H, W)。**必须提供**。

    形状:
        输入 x: (B, N, C)，其中 N = H * W，且 C = heads * dim_head。
        输出: (B, N, C)。

    Example:
        >>> # 示例：dim=128, heads=4, dim_head=32 (4*32=128)
        >>> fact_attn = FactAttention2D(dim=128, heads=4, dim_head=32, shapelist=(32, 32))
        >>> x = torch.randn(2, 32*32, 128)
        >>> out = fact_attn(x)
        >>> out.shape
        torch.Size([2, 1024, 128])
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, shapelist=None):
        super().__init__()
        assert shapelist is not None and len(shapelist) == 2, "FactAttention2D 需要 shapelist=(H, W)"
        assert dim == heads * dim_head, f"Input dim {dim} must equal heads {heads} * dim_head {dim_head}"
        
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.H, self.W = shapelist
        
        self.attn_x = _FactAttnWeight(heads, dim_head, dropout)
        self.attn_y = _FactAttnWeight(heads, dim_head, dropout)
        
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        self.to_x = nn.Sequential(_PoolingReducer(inner_dim, inner_dim, inner_dim))
        
        self.to_y = nn.Sequential(
            Rearrange("b nx ny c -> b ny nx c"),
            _PoolingReducer(inner_dim, inner_dim, inner_dim),
        )
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.H * self.W, f"Sequence length {N} does not match shape {self.H}*{self.W}"

        x = x.reshape(B, self.H, self.W, C).contiguous()
        
        v = (
            self.to_v(
                x.reshape(B, self.H, self.W, self.heads, self.dim_head).contiguous()
            )
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        
        res_x = torch.einsum("bhij,bhjmc->bhimc", self.attn_x(self.to_x(x)), v)
        res_y = torch.einsum("bhlm,bhimc->bhilc", self.attn_y(self.to_y(x)), res_x)
        
        res = rearrange(res_y, "b h i l c -> b (i l) (h c)", h=self.heads)
        return self.to_out(res)


class FactAttention3D(nn.Module):
    """
    3D 因子化注意力 (Factorized Attention 3D)。

    类似于 2D 版本，但处理 3D 体素/网格数据。
    它将注意力分解为 X, Y, Z 三个维度的计算，极大地节省了 3D 数据的显存占用。
    
    **注意**：输入维度 `dim` 必须等于 `heads * dim_head`。

    Args:
        dim (int): 输入特征维度。
        heads (int, optional): 注意力头数。默认值: 8。
        dim_head (int, optional): 每个头的维度。默认值: 64。
        dropout (float, optional): Dropout 概率。默认值: 0.0。
        shapelist (tuple or list): 输入数据的网格形状 (H, W, D)。**必须提供**。

    形状:
        输入 x: (B, N, C)，其中 N = H * W * D，且 C = heads * dim_head。
        输出: (B, N, C)。

    Example:
        >>> # 示例：dim=512, heads=8, dim_head=64 (8*64=512)
        >>> fact_attn = FactAttention3D(dim=512, heads=8, dim_head=64, shapelist=(16, 16, 16))
        >>> x = torch.randn(2, 16**3, 512)
        >>> out = fact_attn(x)
        >>> out.shape
        torch.Size([2, 4096, 512])
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, shapelist=None):
        super().__init__()
        assert shapelist is not None and len(shapelist) == 3, "FactAttention3D 需要 shapelist=(H, W, D)"
        assert dim == heads * dim_head, f"Input dim {dim} must equal heads {heads} * dim_head {dim_head}"
        
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.H, self.W, self.D = shapelist
        
        self.attn_x = _FactAttnWeight(heads, dim_head, dropout)
        self.attn_y = _FactAttnWeight(heads, dim_head, dropout)
        self.attn_z = _FactAttnWeight(heads, dim_head, dropout)
        
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        self.to_x = nn.Sequential(_PoolingReducer(inner_dim, inner_dim, inner_dim))
        
        self.to_y = nn.Sequential(
            Rearrange("b nx ny nz c -> b ny nx nz c"),
            _PoolingReducer(inner_dim, inner_dim, inner_dim),
        )
        
        self.to_z = nn.Sequential(
            Rearrange("b nx ny nz c -> b nz nx ny c"),
            _PoolingReducer(inner_dim, inner_dim, inner_dim),
        )
        
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.H * self.W * self.D, f"Sequence length {N} != {self.H}*{self.W}*{self.D}"
        
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous()
        
        v = (
            self.to_v(
                x.reshape(
                    B, self.H, self.W, self.D, self.heads, self.dim_head
                ).contiguous()
            )
            .permute(0, 4, 1, 2, 3, 5)
            .contiguous()
        )

        res_x = torch.einsum("bhij,bhjmsc->bhimsc", self.attn_x(self.to_x(x)), v)
        res_y = torch.einsum("bhlm,bhimsc->bhilsc", self.attn_y(self.to_y(x)), res_x)
        res_z = torch.einsum("bhrs,bhilsc->bhilrc", self.attn_z(self.to_z(x)), res_y)
        
        res = rearrange(res_z, "b h i l r c -> b (i l r) (h c)", h=self.heads)
        return self.to_out(res)