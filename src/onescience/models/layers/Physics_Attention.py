import torch.nn as nn
import torch
from einops import rearrange, repeat


class Physics_Attention_Irregular_Mesh(nn.Module):
    """
    功能介绍:
        适用于非结构化网格（如点云、有限元离散节点）的物理注意力模块。
        该模块采用“切片-注意力-反切片”（Slice-Attention-Deslice）机制：
        1. 切片（Slice）：通过线性层计算归属度权重，将大量的物理空间点聚合为少量的隐空间“切片Token”。
        2. 注意力（Attention）：在少量的切片Token之间进行标准的多头自注意力计算，捕捉全局特征。
        3. 反切片（Deslice）：利用归属度权重将处理后的切片特征映射回物理空间点。
        这种方法避免了在全量网格点上直接计算注意力，从而将计算复杂度降低到线性水平。

    配置参数:
        dim (int): 输入和输出数据的特征通道数。
        heads (int): 多头注意力的头数。默认值: 8。
        dim_head (int): 每个注意力头的维度大小。默认值: 64。
        dropout (float): Dropout概率，用于防止过拟合。默认值: 0.0。
        slice_num (int): 隐空间切片Token的数量。该数值应远小于网格点数，用于降低计算量。默认值: 64。
        shapelist (list, optional): 在非结构化网格中此参数不被使用，仅为了保持接口统一。
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
    """
    输入数据维度:
        x: (Batch_Size, Num_Points, Channels)
            Batch_Size: 批次大小
            Num_Points: 网格顶点的数量（无序）
            Channels: 输入特征维度 (对应 dim)

    输出数据维度:
        out: (Batch_Size, Num_Points, Channels)
                输出形状与输入保持一致。
    """
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


class Physics_Attention_Structured_Mesh_1D(nn.Module):
    """
    功能介绍:
        适用于一维结构化网格的物理注意力模块。
        与非结构化版本的主要区别在于：在计算切片权重和特征提取时，使用了1D卷积（Conv1d）。
        这使得每个点的信息聚合过程能够感知其局部的空间邻域信息，保留了1D空间的拓扑结构。
        随后同样采用“切片-注意力-反切片”机制进行全局交互。

    配置参数:
        dim (int): 输入和输出数据的特征通道数。
        heads (int): 多头注意力的头数。
        dim_head (int): 每个头的维度。
        dropout (float): Dropout概率。
        slice_num (int): 隐空间切片Token的数量。
        shapelist (list of int): 必须包含一个元素 [Length]，表示1D网格的长度。
        kernel (int): 1D卷积核的大小，用于提取局部特征。默认值: 3。
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
    """
    输入数据维度:
        x: (Batch_Size, Num_Points, Channels)
            注意：虽然输入形式为展平的点序列，但Num_Points必须等于 shapelist[0] (即 Length)。
            模块内部会将其重塑并进行卷积操作。

    输出数据维度:
        out: (Batch_Size, Num_Points, Channels)
    """
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
    功能介绍:
        适用于二维结构化网格（如图像、2D流场）的物理注意力模块。
        利用2D卷积（Conv2d）提取特征并计算切片权重，从而捕捉2D平面的局部空间相关性。
        之后将二维网格压缩为隐空间Token进行全局注意力计算，再还原回二维网格。

    配置参数:
        dim (int): 特征通道数。
        heads (int): 注意力头数。
        dim_head (int): 每个头的维度。
        slice_num (int): 隐空间切片Token的数量。
        shapelist (list of int): 必须包含两个元素 [Height, Width]，定义2D网格的形状。
        kernel (int): 2D卷积核大小。默认值: 3。
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
        """
        输入数据维度:
            x: (Batch_Size, Num_Points, Channels)
               Num_Points 必须等于 Height * Width (即 shapelist乘积)。
               输入是展平的，但在内部会被 reshape 为 (Batch, Channels, Height, Width) 进行处理。

        输出数据维度:
            out: (Batch_Size, Num_Points, Channels)
                 输出是处理后再次展平的序列。
        """
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
    功能介绍:
        适用于三维结构化网格（如3D体数据、气象数据）的物理注意力模块。
        利用3D卷积（Conv3d）在三维空间中提取局部特征并计算切片归属度。
        将巨大的3D体素空间压缩为少量的隐空间Token进行交互，极大降低了3D数据处理的显存占用和计算量。

    配置参数:
        dim (int): 特征通道数。
        heads (int): 注意力头数。
        dim_head (int): 每个头的维度。
        slice_num (int): 隐空间切片Token的数量。通常3D数据点非常多，此参数带来的压缩效果最明显。
        shapelist (list of int): 必须包含三个元素 [Height, Width, Depth]，定义3D网格形状。
        kernel (int): 3D卷积核大小。
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
    """
    输入数据维度:
        x: (Batch_Size, Num_Points, Channels)
            Num_Points 必须等于 Height * Width * Depth (即 shapelist乘积)。
            内部会被 reshape 为 (Batch, Channels, Height, Width, Depth)。

    输出数据维度:
        out: (Batch_Size, Num_Points, Channels)
                输出是处理后再次展平的序列。
    """
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
