import torch
from torch import nn

# from onescience.models.utils import get_earth_position_index, trunc_normal_
from utils import get_earth_position_index, trunc_normal_

from onescience.distributed.megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from onescience.distributed.megatron.core.utils import init_method_normal, scaled_init_method_normal


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        config = None
    ):
        super().__init__()
        self.tp_size = config.tensor_model_parallel_size
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
      
        # sigma = 0.1
        sigma = 0.01
        init_method = init_method_normal(sigma)
        out_init = scaled_init_method_normal(sigma, num_layers=config.num_layers)

        self.fc1 = ColumnParallelLinear(
                input_size = in_features,
                output_size = hidden_features,
                config = config,
                init_method = init_method,
                bias=True
                )
        self.act = act_layer()

        self.fc2 = RowParallelLinear(
                input_size = hidden_features,
                output_size = out_features,
                config = config,
                init_method = out_init,
                bias = True,
                input_is_parallel = True,
                skip_bias_add = False
                )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        #print("before fc1:",x.shape)
        x = self.fc1(x)
        #print("after fc1:",x[0].shape)
        x = self.act(x[0])
        x = self.drop(x)
        x = self.fc2(x)
        #print("after fc2:",x[0].shape)
        x = self.drop(x[0])
        return x


class EarthAttention3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): [pressure levels, latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        config = None
    ):
        super().__init__()
        #print("===para print===")
        #print("dim:",dim)
        #print("input_resolution:",input_resolution)
        #print("window_size:",window_size)
        #print("num_heads:",num_heads)
        #print("qkv_bias:",qkv_bias)
        #print("qk_scale:",qk_scale)
        #print("attn_drop:",attn_drop)
        #print("proj_drop:",proj_drop)
        #print("===para print===")
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        self.tp_size = config.tensor_model_parallel_size
        self.config = config
        assert self.num_heads % self.tp_size == 0 ,"num_heads must be devided by tp_size"
        head_dim = dim // num_heads
        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (
            input_resolution[1] // window_size[1]
        )

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2)
                * (window_size[1] ** 2)
                * (window_size[2] * 2 - 1),
                self.type_of_windows,
                self.num_heads_per_rank,
            )
        )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH
        #print(self.earth_position_bias_table.shape)
        earth_position_index = get_earth_position_index(
            window_size
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        #print("pos index shape:",earth_position_index.shape)
        self.register_buffer("earth_position_index", earth_position_index)

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        sigma = 0.01
        init_method = init_method_normal(sigma)
        out_init = scaled_init_method_normal(sigma, num_layers=config.num_layers)
        self.qkv = ColumnParallelLinear(input_size = dim,
                output_size = dim * 3,
                config = config,
                init_method = init_method,
                bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = RowParallelLinear(
                input_size = dim,
                output_size = dim,
                config = config,
                init_method = out_init,
                bias = True,
                input_is_parallel = True,
                skip_bias_add = False
                )
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=0.01)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_pl*num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape

        x = x.reshape(-1, C)
        qkv, _ = self.qkv(x)
        qkv = qkv.reshape(B_, nW_, N, C*3//self.tp_size)

        qkv = (
            qkv
            .reshape(B_, nW_, N, 3, self.num_heads_per_rank, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        #print("after qvk:",qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = (q * self.scale).float()
        k = k.float()
        v = v.float()

        attn = q @ k.transpose(-2, -1)
        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0).to(attn.dtype)

        if mask is not None:
            nLon = mask.shape[0]
            mask32 = mask.to(torch.float32)

            if mask32.max() > 0:
                mask32 = torch.where(mask32 > 0, torch.full_like(mask32, float("-inf")), torch.zeros_like(mask32))
            attn = attn.view(B_ // nLon, nLon, self.num_heads_per_rank, nW_, N, N) + \
                   mask32.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads_per_rank, nW_, N, N)

        attn = attn - attn.amax(dim=-1, keepdim=True)
        attn = torch.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C // self.tp_size)
        out = out.to(qkv.dtype)

        s1, s2, s3, s4 = out.shape
        out = out.reshape(-1, s4)
        out, _ = self.proj(out)
        out = out.reshape(s1, s2, s3, -1)
        out = self.proj_drop(out)
        return out


class EarthAttention2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [latitude, longitude]
        window_size (tuple[int]): [latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = input_resolution[0] // window_size[0]

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )

        earth_position_index = get_earth_position_index(
            window_size, ndim=2
        )
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=0.01)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_lat, Wlat*Wlon, Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        #print("before qvk:",x.shape)
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        #print("after qvk:",qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = (q * self.scale).float()
        k = k.float()
        v = v.float()

        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.type_of_windows,
            -1,
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0).to(attn.dtype)

        assert torch.isfinite(attn).all(), "non-finite logits before softmax (2D)"

        if mask is not None:
            nLon = mask.shape[0]
            mask32 = mask.to(torch.float32)
            if mask32.max() > 0:
                mask32 = torch.where(mask32 > 0, torch.full_like(mask32, float("-inf")), torch.zeros_like(mask32))
            attn = attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N) + \
                   mask32.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)

        attn = attn - attn.amax(dim=-1, keepdim=True)
        attn = torch.softmax(attn, dim=-1)
        assert torch.isfinite(attn).all(), "non-finite softmax output (2D)"
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        out = out.to(qkv.dtype)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


