"""
Protenix Transformer Modules
Implements transformer blocks for Protenix (AlphaFold3)
Reference: Algorithm 7, 23 in AF3
"""
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.models.openfold.primitives import ProtenixLayerNorm
from onescience.models.protenix.modules.primitives import (
    AdaptiveLayerNorm,
    DropPath,
)
from onescience.modules.linear.protenixlinear import ProtenixLinearNoBias, ProtenixBiasInitLinear
from onescience.modules.attention.protenixattention import ProtenixAttentionPairBiasWithLocalAttn
from onescience.utils.openfold.checkpointing import checkpoint_blocks


class ProtenixConditionedTransitionBlock(nn.Module):
    """
    Implements Algorithm 25 in AF3
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.n = n
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = ProtenixLinearNoBias(in_features=c_a, out_features=n * c_a, initializer="relu")
        self.linear_nobias_a2 = ProtenixLinearNoBias(in_features=c_a, out_features=n * c_a, initializer="relu")
        self.linear_nobias_b = ProtenixLinearNoBias(in_features=n * c_a, out_features=c_a)
        self.linear_s = ProtenixBiasInitLinear(
            in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.adaln(a, s)
        b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
        a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
        return a


class ProtenixDiffusionTransformerBlock(nn.Module):
    """
    Implements Algorithm 23[Line2-Line3] in AF3
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        n_heads: int,
        biasinit: float = -2.0,
        drop_path_rate: float = 0.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.attention_pair_bias = ProtenixAttentionPairBiasWithLocalAttn(
            has_s=True,
            create_offset_ln_z=False,
            n_heads=n_heads,
            c_a=c_a,
            c_s=c_s,
            c_z=c_z,
            biasinit=biasinit,
            cross_attention_mode=cross_attention_mode,
        )
        self.conditioned_transition_block = ProtenixConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        attn_out = self.drop_path(
            self.attention_pair_bias(
                a=a,
                s=s,
                z=z,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        )
        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a
        ff_out = self.conditioned_transition_block(a=attn_out, s=s)
        out_a = ff_out + attn_out
        return out_a, s, z


class ProtenixDiffusionTransformer(nn.Module):
    """
    Implements Algorithm 23 in AF3
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        n_blocks: int,
        n_heads: int,
        cross_attention_mode: bool = False,
        drop_path_rate: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        drop_path_rates = [
            drop_path_value.item()
            for drop_path_value in torch.linspace(0, drop_path_rate, n_blocks)
        ]
        for i in range(n_blocks):
            block = ProtenixDiffusionTransformerBlock(
                n_heads=n_heads,
                c_a=c_a,
                c_s=c_s,
                c_z=c_z,
                cross_attention_mode=cross_attention_mode,
                drop_path_rate=drop_path_rates[i],
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ):
        blocks = [
            partial(
                b,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False
        blocks = self._prep_blocks(
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        a, s, z = checkpoint_blocks(
            blocks, args=(a, s, z), blocks_per_ckpt=blocks_per_ckpt
        )
        del s, z
        return a


class ProtenixAtomTransformer(nn.Module):
    """
    Implements Algorithm 7 in AF3
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.diffusion_transformer = ProtenixDiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atompair,
            cross_attention_mode=True,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        n_blocks, n_queries, n_keys = p.shape[-4:-1]

        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

