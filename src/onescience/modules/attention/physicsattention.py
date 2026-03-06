import torch.nn as nn
import torch
from einops import rearrange, repeat
from timm.layers import trunc_normal_
from einops import rearrange
import torch.distributed.nn as dist_nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.distributed as dist

def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y

class Physics_Attention_Irregular_Mesh(nn.Module):
    """
        适用于非结构化网格（如点云、有限元离散节点）的物理注意力模块。

        该模块采用了 Slice-Attention-Deslice（切片-注意力-反切片）机制，旨在降低在大量网格点上进行全局注意力计算的复杂度。
        过程包含：
        1. Slice: 通过可学习的线性映射计算归属度权重，将物理空间中大量的点聚合为少量的隐空间“切片 Token”。
        2. Attention: 在这些少量的切片 Token 之间执行标准的多头自注意力，捕捉全局物理特征。
        3. Deslice: 利用归属度权重将处理后的隐空间特征映射回原始的物理空间点。
        这种方法避免了 O(N^2) 的全量注意力计算，实现了线性的计算复杂度。

        Args:
            dim (int): 输入和输出数据的特征通道数
            heads (int, optional): 多头注意力的头数，默认为 8
            dim_head (int, optional): 每个注意力头的维度大小，默认为 64
            dropout (float, optional): Dropout 概率，默认为 0.0
            slice_num (int, optional): 隐空间切片 Token 的数量。该数值通常远小于网格点数 N，用于控制压缩比和计算量，默认为 64
            shapelist (list, optional): 仅为了保持接口统一，在此模块中未被使用，默认为 None

        形状:
            输入 x: (B, N, C)，其中 B 为批次大小，N 为无序网格点数，C 为特征维度
            输出: (B, N, C)，输出形状与输入保持一致

        Example:
            >>> # 假设有一组包含 2048 个点的点云数据
            >>> attn = Physics_Attention_Irregular_Mesh(dim=128, slice_num=64)
            >>> x = torch.randn(8, 2048, 128)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([8, 2048, 128])

    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

class Physics_Attention_Irregular_Mesh_plus(nn.Module):
    """
        增强型非结构化网格物理注意力模块（集成动态温度调节与分布式支持）。

        该模块是 Physics_Attention_Irregular_Mesh 的升级版本。主要特性包括：
        1. 动态温度调节 (Eidetic): 通过辅助网络根据输入特征动态预测 Gumbel Softmax 的温度参数，自适应调整聚类的“锐度”。
        2. Gumbel Softmax: 替代标准 Softmax，支持更具随机性和离散性质的聚类分配。
        3. 分布式支持: 内置 dist.all_reduce，支持跨 GPU 同步隐空间切片特征，确保全局物理场一致性。
        4. 高效计算: 使用 scaled_dot_product_attention 进行加速。

        Args:
            dim (int): 输入和输出数据的特征通道数
            heads (int, optional): 多头注意力的头数，默认为 8
            dim_head (int, optional): 每个注意力头的维度大小，默认为 64
            dropout (float, optional): Dropout 概率，默认为 0.0
            slice_num (int, optional): 隐空间切片 Token 的数量，默认为 64
            shapelist (list, optional): 仅为了保持接口统一，在此模块中未被使用，默认为 None

        形状:
            输入 x: (B, N, C)，其中 B 为批次大小，N 为无序网格点数，C 为特征维度
            输出: (B, N, C)，输出形状与输入保持一致

        Example:
            >>> # 假设分布式环境或单机，处理 4096 个无序物理点
            >>> model = Physics_Attention_Irregular_Mesh_plus(dim=128, slice_num=64)
            >>> x = torch.randn(2, 4096, 128)
            >>> out = model(x)
            >>> out.shape
            torch.Size([2, 4096, 128])

    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # B N C
        B, N, C = x.shape

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)  # B H G
        if dist.is_available() and dist.is_initialized():
            dist_nn.all_reduce(slice_norm, op=dist_nn.ReduceOp.SUM)            
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        if dist.is_available() and dist.is_initialized():
            dist_nn.all_reduce(slice_token, op=dist_nn.ReduceOp.SUM)          
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

class Physics_Attention_Structured_Mesh_1D(nn.Module):
    """
        适用于一维结构化网格的物理注意力模块。

        与非结构化版本不同，该模块在特征提取阶段利用 1D 卷积 (Conv1d) 来提取局部特征并计算切片权重。
        这使得每个点在聚合到隐空间之前，能够感知其在 1D 拓扑结构中的局部邻域信息。
        随后同样采用 Slice-Attention-Deslice 机制进行全局交互。

        Args:
            dim (int): 输入和输出数据的特征通道数
            heads (int, optional): 多头注意力的头数，默认为 8
            dim_head (int, optional): 每个注意力头的维度大小，默认为 64
            dropout (float, optional): Dropout 概率，默认为 0.0
            slice_num (int, optional): 隐空间切片 Token 的数量，默认为 64
            shapelist (list[int]): 必须包含一个元素 [Length]，指定一维网格的长度。输入张量的点数 N 必须等于 Length
            kernel (int, optional): 用于提取局部特征的 1D 卷积核大小，默认为 3

        形状:
            输入 x: (B, N, C)，注意内部会将 N 重塑为 shapelist 指定的长度 Length
            输出: (B, N, C)，输出形状与输入保持一致

        Example:
            >>> # 1D 序列长度为 100
            >>> attn = Physics_Attention_Structured_Mesh_1D(dim=64, shapelist=[100])
            >>> x = torch.randn(10, 100, 64)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([10, 100, 64])

    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None, kernel=3):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.length = shapelist[0]

        self.in_project_x = nn.Conv1d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv1d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.length, C).contiguous().permute(0, 2, 1).contiguous()  # B C N

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    """
        适用于二维结构化网格（如图像、2D流场）的物理注意力模块。

        该模块在特征提取阶段使用 2D 卷积 (Conv2d)。
        它首先将展平的输入序列重塑为 2D 网格，利用卷积捕捉 2D 平面上的局部空间相关性，
        然后通过切片机制压缩至隐空间进行全局注意力计算，最后还原回物理空间。

        Args:
            dim (int): 输入和输出数据的特征通道数
            heads (int, optional): 多头注意力的头数，默认为 8
            dim_head (int, optional): 每个注意力头的维度大小，默认为 64
            dropout (float, optional): Dropout 概率，默认为 0.0
            slice_num (int, optional): 隐空间切片 Token 的数量，默认为 64
            shapelist (list[int]): 必须包含两个元素 [Height, Width]，指定二维网格的形状。输入张量的点数 N 必须满足 N = H * W
            kernel (int, optional): 用于提取局部特征的 2D 卷积核大小，默认为 3

        形状:
            输入 x: (B, N, C)，其中 N = H * W
            输出: (B, N, C)，输出形状与输入保持一致

        Example:
            >>> # 64x64 的 2D 网格，总点数 N=4096
            >>> attn = Physics_Attention_Structured_Mesh_2D(dim=32, shapelist=[64, 64], slice_num=128)
            >>> x = torch.randn(4, 4096, 32)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([4, 4096, 32])

    """
    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = shapelist[0]
        self.W = shapelist[1]

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    """
        适用于三维结构化网格（如 3D 体数据、气象数据）的物理注意力模块。

        该模块专门针对高维数据设计，使用 3D 卷积 (Conv3d) 在三维空间中提取局部特征并计算切片归属度。
        通过将巨大的 3D 体素空间压缩为极少量的隐空间 Token 进行交互，该模块能够显著降低 3D 数据处理时的显存占用和计算量。

        Args:
            dim (int): 输入和输出数据的特征通道数
            heads (int, optional): 多头注意力的头数，默认为 8
            dim_head (int, optional): 每个注意力头的维度大小，默认为 64
            dropout (float, optional): Dropout 概率，默认为 0.0
            slice_num (int, optional): 隐空间切片 Token 的数量。对于 3D 数据，此参数带来的压缩效果通常最为明显，默认为 32
            shapelist (list[int]): 必须包含三个元素 [Height, Width, Depth]，指定三维网格的形状。输入张量的点数 N 必须满足 N = H * W * D
            kernel (int, optional): 用于提取局部特征的 3D 卷积核大小，默认为 3

        形状:
            输入 x: (B, N, C)，其中 N = H * W * D
            输出: (B, N, C)，输出形状与输入保持一致

        Example:
            >>> # 32x32x32 的 3D 体数据，总点数 N=32768
            >>> attn = Physics_Attention_Structured_Mesh_3D(dim=16, shapelist=[32, 32, 32], slice_num=64)
            >>> x = torch.randn(2, 32768, 16)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([2, 32768, 16])

    """
    ## for structured mesh in 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, shapelist=None, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = shapelist[0]
        self.W = shapelist[1]
        self.D = shapelist[2]

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
