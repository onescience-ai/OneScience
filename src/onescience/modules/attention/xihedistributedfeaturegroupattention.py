import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp
from onescience.distributed.megatron.core.tensor_parallel.layers import RowParallelLinear
from onescience.distributed.megatron.core.utils import init_method_normal, scaled_init_method_normal


class DistributedFeatureGroupingAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_groups=32,
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
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.config = config

        self.group_vectors = nn.Parameter(torch.randn(1, num_groups, dim))
        self.norm = LN(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, bias=qkv_bias, batch_first=True
        )
        self.attn_drop = drop_layer(attn_drop)

        sigma = 0.01
        init_method = init_method_normal(sigma)
        out_init = scaled_init_method_normal(sigma, num_layers=config.num_layers)
        self.proj = RowParallelLinear(
            input_size=dim,
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
            x = obj["x"]
            mask_tokens = obj.get("mask")
            if mask_tokens is not None:
                mask_tokens = mask_tokens.clone().detach().float()
        else:
            x = obj.x
            mask_tokens = obj.mask
            obj = {"x": x, "mask": mask_tokens}

        B, N, C = x.shape
        x = self.norm(x)

        G = self.group_vectors.expand(B, -1, -1)

        if mask_tokens is None:
            return None
        if mask_tokens.dim() == 4:
            mask_tokens = mask_tokens.squeeze(1)
        if mask_tokens.dim() == 3:
            mask_tokens = mask_tokens.reshape(B, -1)
        assert mask_tokens.shape == (B, N)
        key_padding_mask = None if mask_tokens is None else (mask_tokens == 0)
        G_prime, _ = self.attn(query=G, key=x, value=x, key_padding_mask=key_padding_mask)

        G_prime = G_prime.reshape(-1, C)
        G_prime, _ = self.proj(G_prime)
        G_prime = G_prime.reshape(B, -1, C)
        G_prime = self.proj_drop(G_prime)

        return G_prime
