from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..layers.transformer_layers import Transformer3DBlock
from ..layers.mlp_layers import Mlp
from collections.abc import Sequence
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from onescience.models.xihe.localsie import LocalSIE
from onescience.models.xihe.globalsie import GlobalSIE

from ..utils import (
    PatchEmbed2D,
    PatchRecovery2D,
    crop2d,
    crop3d,
    get_pad2d,
    get_pad3d,
    get_shift_window_mask,
    window_partition,
    window_reverse,
)
from ..layers.attention_layers import EarthAttention2D, EarthAttention3D
from ..layers.drop import DropPath
from ..layers.mlp_layers import Mlp
from ..layers.resample_layers import DownSample2D, UpSample2D



# class Transformer3DBlock(nn.Module):
#     """
#     Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
#     3D Transformer Block
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
#         shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(
#         self,
#         dim,
#         input_resolution,
#         num_heads,
#         window_size=None,
#         shift_size=None,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         window_size = (2, 6, 12) if window_size is None else window_size
#         shift_size = (1, 3, 6) if shift_size is None else shift_size
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         self.norm1 = norm_layer(dim)
#         padding = get_pad3d(input_resolution, window_size)
#         self.pad = nn.ZeroPad3d(padding)
#         attn_mask=None

#         pad_resolution = list(input_resolution)
#         pad_resolution[0] += padding[-1] + padding[-2]
#         pad_resolution[1] += padding[2] + padding[3]
#         pad_resolution[2] += padding[0] + padding[1]

#         self.attn = EarthAttention3D(
#             dim=dim,
#             input_resolution=pad_resolution,
#             window_size=window_size,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )

#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )

#         shift_pl, shift_lat, shift_lon = self.shift_size
#         self.roll = shift_pl and shift_lon and shift_lat

#         if self.roll:
#             attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
#         else:
#             attn_mask = None

#         self.register_buffer("attn_mask", attn_mask)
        

#     def forward(self, x: torch.Tensor,mask: torch.Tensor = None):
#         Pl, Lat, Lon = self.input_resolution
#         B, L, C = x.shape

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, Pl, Lat, Lon, C)
#         # print("mask",mask)
#         # start pad
#         x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

#         _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

#         shift_pl, shift_lat, shift_lon = self.shift_size
        
#         if self.roll:
#             shifted_x = torch.roll(
#                 x, shifts=(-shift_pl, -shift_lat, -shift_lat), dims=(1, 2, 3)
#             )
#             x_windows = window_partition(shifted_x, self.window_size)
#             # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
#         else:        
#             shifted_x = x
#             x_windows = window_partition(shifted_x, self.window_size)
#             # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
#         win_pl, win_lat, win_lon = self.window_size
        
#         x_windows = x_windows.view(
#             x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
#         )
 
#         attn_mask = None
#         if mask is not None:
#             # 期望 mask 是 [B, 1, Lat, Lon] 或 [B, 1, Pl, Lat, Lon]
#             if mask.dim() == 4:                # (B,1,Lat,Lon) -> (B,1,1,Lat,Lon)
#                 mask = mask.unsqueeze(2)

#             # 此时 mask: (B, 1, Pl, Lat, Lon)
#             #  期望 (N, C, D, H, W)；这里 C=1, D=Pl, H=Lat, W=Lon，直接 pad 即可
#             mask = self.pad(mask)              # (B, 1, Pl_pad, Lat_pad, Lon_pad)

#             # 为了与 window_partition 通用实现对齐，转成 (B, Pl_pad, Lat_pad, Lon_pad, 1)
#             mask5d = mask.permute(0, 2, 3, 4, 1).contiguous()

#             # 与特征 x 完全一致的分块（3D窗口）
#             # mwin: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1)
#             mwin = window_partition(mask5d, self.window_size)

#             win_pl, win_lat, win_lon = self.window_size
#             # 计算分块数量
#             # 注意：x 已经 pad 过，这里的 Pl_pad/Lat_pad/Lon_pad 要和上面 x 的 pad 后维度一致
#             _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape               # x 此时是 pad 后的 (B, Pl_pad, Lat_pad, Lon_pad, C)
#             B_eff  = mask5d.shape[0]
#             num_lon   = Lon_pad // win_lon
#             num_pllat = (Pl_pad // win_pl) * (Lat_pad // win_lat)
#             N = win_pl * win_lat * win_lon                         # 每个窗口 token 数

#             # 把 (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1) 还原出 (B, num_lon, num_pl*num_lat, N)
#             mwin = mwin.view(B_eff, num_lon, num_pllat, win_pl, win_lat, win_lon, 1)
#             # 取第 0 个 batch
#             mwin = mwin[0]                                         # (num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1)
#             mwin = mwin.view(num_lon, num_pllat, N)                # (num_lon, num_pl*num_lat, N)，元素∈{0,1}

#             # 生成注意力掩码 (num_lon, num_pl*num_lat, N, N) 仅允许 海×海，其他（涉及陆地）设为 -inf
#             attn_mask = (mwin.unsqueeze(-1) * mwin.unsqueeze(-2))  # 0/1
#             # ratio = (attn_mask == 1).float().mean().item()
#             # print("海洋区域占比:", ratio)
#             # print("mask1",attn_mask.shape,x.shape)
#             attn_mask = (attn_mask == 0).float() * -100.0          # 变成 0 / -100



#         attn_windows = self.attn(x_windows, mask=attn_mask)
#         attn_windows = attn_windows.view(
#             attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C
#         )

#         if self.roll:
#             shifted_x = window_reverse(
#                 attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
#             )
#             # B * Pl * Lat * Lon * C
#             x = torch.roll(
#                 shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3)
#             )
#         else:
#             shifted_x = window_reverse(
#                 attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
#             )
#             x = shifted_x

#         # crop, end pad
#         x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(
#             0, 2, 3, 4, 1
#         )

#         x = x.reshape(B, Pl * Lat * Lon, C)
#         #两次残差
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x


# class LocalSIE(nn.Module):
#     """Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
#     A basic 3D Transformer layer for one stage

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (tuple[int]): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#     """

#     def __init__(
#         self,
#         dim,
#         input_resolution,
#         depth,
#         num_heads,
#         window_size,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth

#         self.blocks = nn.ModuleList(
#             [
#                 Transformer3DBlock(
#                     dim=dim,
#                     input_resolution=input_resolution, #3d windows
#                     num_heads=num_heads,
#                     window_size=window_size,
#                     shift_size=(0, 0, 0), #不让他选择swin 
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop,
#                     attn_drop=attn_drop,
#                     drop_path=drop_path[i]
#                     if isinstance(drop_path, Sequence)
#                     else drop_path,
#                     norm_layer=norm_layer,
#                 )
#                 for i in range(depth)
#             ]
#         )

#     def forward(self, x,mask=None):
#         for blk in self.blocks:
#             x = blk(x) if mask is None else blk(x,mask=mask)
#         return x

# #GLOBAL 1
# class FeatureGrouping(nn.Module):
#     """
#     Global SIE - Step 1: Feature Grouping (论文公式 (5) 实现)
#     --------------------------------------------------------
#     输入: Z_tilde (B, N, C)  # Local SIE 输出的特征
#     输出: G_prime (B, G, C) # 更新后的 group vectors
#     """

#     def __init__(
#         self,
#         dim, 
#         num_groups=32, 
#         num_heads=8, 
#         qkv_bias=True,
#         attn_drop=0.0, 
#         proj_drop=0.0,
#         LN=nn.LayerNorm,
#         drop_layer=nn.Dropout,
#         ):
#         super().__init__()
#         self.dim = dim
#         self.num_groups = num_groups  
#         self.num_heads = num_heads  #自定义，分组多就可表示的更精细

#         # 初始化 learnable group vectors (相当于 G_l)
#         self.group_vectors = nn.Parameter(torch.randn(1, num_groups, dim))
#         # 1.LN 作用在输入 patch 特征 Z_tilde 上
#         self.norm = LN(dim)
#         # 2.多头注意力 (标准 vanilla Transformer Attention)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim, num_heads=num_heads, bias=qkv_bias, batch_first=True
#         )
#         self.attn_drop = drop_layer(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = drop_layer(proj_drop)
#     def forward(self, x,mask_tokens=None):
#         """
#         x: (B, N, C)  -> 来自 Local SIE 的特征
#         """
#         B, N, C = x.shape

#         # 1. 归一化输入特征
#         x = self.norm(x)  # (B, N, C)
        
#         # 2. expand group vectors (batch 内共享同一份 group 参数)
#         G = self.group_vectors.expand(B, -1, -1)  # (B, G, C)
#         # 3. Multi-Head Cross-Attention
#         if mask_tokens is None:
#             return None
#         if mask_tokens.dim() == 4:              # (B,1,H,W)
#             mask_tokens = mask_tokens.squeeze(1)
#         if mask_tokens.dim() == 3:              # (B,H,W)
#             mask_tokens = mask_tokens.reshape(B, -1)
#         assert mask_tokens.shape == (B, N)
#         key_padding_mask = (mask_tokens == 0)   # True=忽略
#         # 统计海洋与陆地占比
#         # land_ratio = key_padding_mask.float().mean().item()         # 陆地(忽略)=True
#         # ocean_ratio = 1.0 - land_ratio                              # 海洋(保留)
#         # print(f"[DEBUG] 掩码统计: 海洋占比={ocean_ratio:.3f}, 陆地占比={land_ratio:.3f}")
#         # mask_tokens: (B,N) 或 (B,1,H',W')/(B,H',W')，1=海洋, 0=陆地  
#         #    Q = G,  K,V = x  （加入掩码，屏蔽掉陆地）
#         G_prime, _ = self.attn(query=G, key=x, value=x,key_padding_mask=key_padding_mask)
#         # 4. 输出更新后的 group vectors
#         G_prime = self.proj_drop(self.proj(G_prime))  # (B, G, C)

#         return G_prime

# #GLOBAL 2   
# class GroupPropagation(nn.Module):
#     """
#     Global SIE - Step 2: Group Propagation (论文公式 (6)(7) 实现)
#     ------------------------------------------------------------
#     输入:  G_prime (B, G, C)   # 来自 Feature Grouping 的 group vectors
#     输出:  G_tilde (B, G, C)   # 融合全局信息后的 group vectors
#     """

#     def __init__(
#         self,
#         dim, 
#         num_groups,
#         mlp_ratio=4.0,
#         drop=0.0, 
#         act_layer=nn.GELU,
#         LN=nn.LayerNorm
#         ):
#         super().__init__()
#         self.dim = dim
#         self.num_groups = num_groups

#         # LayerNorm
#         self.norm1 = LN(dim)
#         self.norm2 = LN(dim)
        
#         # 111  Token-mixing MLP (在 group 维度上传播信息)
#         mlp_token_dim = int(num_groups * mlp_ratio)
#         self.mlp_token = Mlp(
#             in_features=num_groups,
#             hidden_features=mlp_token_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )        
    
#        # 222 Channel-mixing MLP (在 embedding 维度融合特征)
#         mlp_channel_dim = int(dim * mlp_ratio)
#         self.mlp_channel =Mlp(
#             in_features=dim,
#             hidden_features=mlp_channel_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )
        
#     def forward(self, x):
#         """
#         x: (B, G, C) -> 输入 group vectors
#         """
#         B, G, C = x.shape
#         shortcut=x
        
#         # print("x.shape",x.shape)
#         # Step 1: Token mixing (group 维度传播信息)
#         x = self.norm1(x)          # (B, G, C)
#         x = x.transpose(1, 2)            # (B, C, G)
#         x = self.mlp_token(x)            # (B, C, G) 先对group进行mlp
#         x = x.transpose(1, 2)            # (B, G, C)
#         x = shortcut + x              # 残差连接
#         # Step 2: Channel mixing (embedding 维度融合)  
#         y = self.norm2(x)            # (B, G, C)
#         y = self.mlp_channel(y)          # (B, G, C) 在对channel进行mlp
#         y = x + y              # 残差连接
#         return y

# #GLOBAL 3
# class FeatureUngrouping(nn.Module):
#     """
#     Global SIE - Step 3: Feature Ungrouping
#     --------------------------------------------------------
#     输入: 
#       - x: (B, N, C) patch tokens (来自 Local SIE 输出)
#       - G_tilde: (B, G, C) group vectors (经过 Group Propagation)
#     输出: 
#       - x_out: (B, N, C) 融合全局信息的 patch tokens
#     """

#     def __init__(
#         self,
#         dim,
#         num_heads=8,
#         qkv_bias=True,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         LN=nn.LayerNorm,
#         drop_layer=nn.Dropout,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
        
     
#         self.norm_x = LN(dim)  # 对 patch tokens 做归一化
#         self.norm_g = LN(dim)  # 对 group vectors 做归一化
        
        

#         # Cross-Attention (Q=patch tokens, K/V=group vectors)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop,batch_first=True
#         )
        
#         # 注意力输出的投影层
#         self.attn_proj = nn.Linear(dim, dim)
#          # 拼接后的融合层
#         self.concat_proj = nn.Linear(2 * dim, dim)
#         self.proj_drop = drop_layer(proj_drop)
        
#         # self.attn_drop = drop_layer(attn_drop)
#         # self.proj_drop = drop_layer(proj_drop)

#     def forward(self, x, G_tilde,mask=None):
#         """
#         x: (B, N, C)  patch tokens
#         G_tilde: (B, G, C)  group vectors
#         """
#         B, N, C = x.shape
#         _, G, _ = G_tilde.shape

#         # 归一化
#         x_norm = self.norm_x(x)
#         G_norm = self.norm_g(G_tilde)

#         # Cross-Attention: Q = x, K/V = groups
#         x_out, _ = self.attn(query=x_norm, key=G_norm, value=G_norm)
#         x_out = self.proj_drop(self.attn_proj(x_out))
        
#         # 拼接 [U, x] 并线性映射回原维度 C
#         x_concat = torch.cat([x_out, x], dim=-1)   # (B, N, 2C)
        
#         # print("x_concat:",x_concat.shape,B,N,C)
        
#         # x_out = self.concat_proj(x_concat)    
#         x_out = self.proj_drop(self.concat_proj(x_concat))  # (B, N, C)

#         # 残差连接：原始 patch + 全局信息
#         # x_out = x + x_out

#         return x_out

# class GlobalSIE(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         qkv_bias=True,
#         num_groups=32,
#         norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.dim=dim
#         self.num_heads=num_heads
#         self.num_group=num_groups

        
#         self.feature_grouping = FeatureGrouping(
#             dim=dim,
#             num_groups=num_groups,       # 超参数，可调整，靠输入数据决定
#             num_heads=num_heads,         # 和 local 对齐
#             qkv_bias=qkv_bias,
            
#         )
#         self.group_propagation = GroupPropagation(
#             dim=dim,
#             num_groups=num_groups,                
#             mlp_ratio=4.0,               # 和 local 对齐
#             drop=0.0,                    # 和 local 对齐
#             act_layer=nn.GELU           
#         ) 
#         self.feature_ungrouping = FeatureUngrouping(        
#             dim=dim,
#             num_heads=num_heads,         # 和 local 对齐
#             qkv_bias=qkv_bias           # 和 local 对齐            
#         )

#     def forward(self, x,mask=None):
#         y=x
#         x=self.feature_grouping(x,mask_tokens=mask)
#         x=self.group_propagation(x)
#         x=self.feature_ungrouping(y,x)
 
#         return x
        
class OceanSpecificBlock(nn.Module):
    """
    Ocean-Specific Block
    ---------------------
    Block1 & Block5: 1 Local + 1 Global
    Block2-Block4 : 2 Local + 1 Global
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_local,
        num_heads_global,
        window_size,
        mlp_ratio,
        qkv_bias=True,
        drop_path=0.0,
        num_groups=32,
        num_local=1,        #  Number of Local SIE 
        num_global=1,       #  Number of Global SIE
        depth_local=2,      #  depth of transformer block
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim=dim
        self.num_groups=num_groups
        self.num_local=num_local
        self.num_global=num_global
        self.num_heads_local=num_heads_local    
        self.num_heads_global=num_heads_global
        self.window_size=window_size
        self.drop_path=drop_path

        # Local SIE modules
        self.local_sie_blocks = nn.ModuleList([
            LocalSIE(
                dim=dim,
                input_resolution=input_resolution,
                depth=depth_local,
                num_heads=num_heads_local,
                window_size=window_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=drop_path,
                norm_layer=norm_layer,                
            )
            for _ in range(num_local)
        ])

        # Global SIE modules
        self.global_sie_blocks = nn.ModuleList([
            GlobalSIE(
                dim=dim,
                num_heads=num_heads_global,
                num_groups=num_groups,
                norm_layer=norm_layer,
            )
            for _ in range(num_global)
        ])

    def forward(self, x, mask=None):
        """
        x: (B, N, C)
        mask: (可选) ocean-land mask
        """
        # Local SIE(s)
        for local in self.local_sie_blocks:
            x = local(x) if mask is None else local(x, mask=mask)

        # Global SIE(s)
        for global_sie in self.global_sie_blocks:
            x = global_sie(x) if mask is None else global_sie(x, mask=mask)

        return x
    



    