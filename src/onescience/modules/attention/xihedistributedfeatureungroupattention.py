import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp
from onescience.distributed.megatron.core.tensor_parallel.layers import RowParallelLinear
from onescience.distributed.megatron.core.utils import init_method_normal, scaled_init_method_normal


class XiheDistributedFeatureUngroupingAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        LN=nn.LayerNorm,
        drop_layer=nn.Dropout,
        config=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.config = config

        self.norm_x = LN(dim)
        self.norm_g = LN(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop, batch_first=True
        )

        sigma = 0.01
        init_method = init_method_normal(sigma)
        out_init = scaled_init_method_normal(sigma, num_layers=config.num_layers)

        self.attn_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            config=config,
            init_method=out_init,
            bias=True,
            input_is_parallel=False,
            skip_bias_add=False,
        )
        self.concat_proj = RowParallelLinear(
            input_size=2 * dim,
            output_size=dim,
            config=config,
            init_method=out_init,
            bias=True,
            input_is_parallel=False,
            skip_bias_add=False,
        )
        self.proj_drop = drop_layer(proj_drop)

    def forward(self, obj, mask=None):
        if isinstance(obj, dict):
            x = obj["y"]
            G_tilde = obj["x"]
        else:
            x = obj.y
            G_tilde = obj.x

        B, N, C = x.shape
        _, G, _ = G_tilde.shape

        x_norm = self.norm_x(x)
        G_norm = self.norm_g(G_tilde)

        x_out, _ = self.attn(query=x_norm, key=G_norm, value=G_norm)

        x_out = x_out.reshape(-1, C)
        x_out, _ = self.attn_proj(x_out)
        x_out = x_out.reshape(B, N, C)
        x_out = self.proj_drop(x_out)

        x_concat = torch.cat([x_out, x], dim=-1)

        x_concat = x_concat.reshape(-1, 2 * C)
        x_out, _ = self.concat_proj(x_concat)
        x_out = x_out.reshape(B, N, C)
        x_out = self.proj_drop(x_out)

        return x_out
