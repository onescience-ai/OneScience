"""Protenix attention modules for AlphaFold3.

This module implements various attention mechanisms including standard multi-head
attention and attention with pair bias, as described in Algorithm 24 of AlphaFold3.
"""
import math
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.models.openfold.primitives import ProtenixLayerNorm
from onescience.models.protenix.modules.primitives import (
    AdaptiveLayerNorm,
    create_local_attn_bias,
    _local_attention,
)
from onescience.models.protenix.utils import (
    permute_final_dims,
    flatten_final_dims,
)
from onescience.modules.linear.protenixlinear import (
    ProtenixLinear,
    ProtenixLinearNoBias,
    ProtenixBiasInitLinear,
)


def _protenix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    use_efficient_implementation: bool = False,
    inplace_safe: bool = False,
) -> torch.Tensor:
    """Computes standard scaled dot-product attention.

    Args:
        q: Query tensor. Shape: [..., n_q, d].
        k: Key tensor. Shape: [..., n_kv, d].
        v: Value tensor. Shape: [..., n_kv, d].
        attn_bias: Optional attention bias. Shape: [..., n_q, n_kv].
        use_efficient_implementation: Whether to use PyTorch's efficient SDPA.
            Defaults to False.
        inplace_safe: Whether inplace operations are safe. Defaults to False.

    Returns:
        Attention output. Shape: [..., n_q, d].
    """
    assert k.shape == v.shape

    input_dtype = q.dtype
    q = q.to(dtype=torch.float32)
    k = k.to(dtype=torch.float32)
    if attn_bias is not None:
        attn_bias = attn_bias.to(dtype=torch.float32)

    if use_efficient_implementation:
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_bias,
        )
        return attn_output

    with torch.cuda.amp.autocast(enabled=False):
        k = k.transpose(-1, -2)
        attn_weights = q @ k

        if attn_bias is not None:
            if inplace_safe:
                attn_weights += attn_bias
            else:
                attn_weights = attn_weights + attn_bias

        attn_weights = F.softmax(attn_weights, dim=-1)

    attn_output = attn_weights.to(dtype=input_dtype) @ v
    return attn_output


class ProtenixAttention(nn.Module):
    """Standard multi-head attention for Protenix.

    Implements multi-head attention with optional gating and local attention support.
    Reference: OpenFold implementation (https://github.com/aqlaboratory/openfold).
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        gating: bool = True,
        q_linear_bias: bool = True,
        local_attention_method: str = "global_attention_with_bias",
        use_efficient_implementation: bool = False,
        zero_init: bool = True,
    ) -> None:
        """Initializes the ProtenixAttention module.

        Args:
            c_q: Input dimension of query.
            c_k: Input dimension of key.
            c_v: Input dimension of value.
            c_hidden: Per-head hidden dimension.
            num_heads: Number of attention heads.
            gating: Whether to use gating mechanism. Defaults to True.
            q_linear_bias: Whether to use bias in query projection. Defaults to True.
            local_attention_method: Method for local attention ('global_attention_with_bias'
                or 'local_cross_attention'). Defaults to 'global_attention_with_bias'.
            use_efficient_implementation: Whether to use PyTorch's efficient SDPA.
                Defaults to False.
            zero_init: Whether to zero-initialize output layer. Defaults to True.
        """
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.gating = gating
        self.local_attention_method = local_attention_method
        self.use_efficient_implementation = use_efficient_implementation

        if q_linear_bias:
            self.linear_q = ProtenixLinear(
                in_features=self.c_q, out_features=self.c_hidden * self.num_heads
            )
        else:
            self.linear_q = ProtenixLinearNoBias(self.c_q, self.c_hidden * self.num_heads)
        self.linear_k = ProtenixLinearNoBias(self.c_k, self.c_hidden * self.num_heads)
        self.linear_v = ProtenixLinearNoBias(self.c_v, self.c_hidden * self.num_heads)
        self.linear_o = ProtenixLinearNoBias(self.c_hidden * self.num_heads, self.c_q)
        self.linear_g = None
        if self.gating:
            self.linear_g = ProtenixLinearNoBias(
                self.c_q, self.c_hidden * self.num_heads, initializer="zeros"
            )
            self.sigmoid = nn.Sigmoid()

        self.zero_init = zero_init
        if self.zero_init:
            nn.init.zeros_(self.linear_o.weight)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare Q, K, V tensors."""
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        """Wrap up attention output with gating."""
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.num_heads, -1))
            o = o * g

        o = flatten_final_dims(o, num_dims=2)
        o = self.linear_o(o)
        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        trunked_attn_bias: Optional[torch.Tensor] = None,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inf: Optional[float] = 1e10,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Computes multi-head attention with optional local attention.

        Args:
            q_x: Query input. Shape: [..., Q, c_q].
            kv_x: Key/Value input. Shape: [..., K, c_k].
            attn_bias: Attention bias. Shape: [..., Q, K] or [..., H, Q, K].
            trunked_attn_bias: Trunked attention bias for local attention.
                Shape: [..., H, n_trunks, n_queries, n_keys].
            n_queries: Local window size for queries. If None, global attention is used.
            n_keys: Local window size for keys. If None, global attention is used.
            inf: Infinity value for masking. Defaults to 1e10.
            inplace_safe: Whether inplace operations are safe. Defaults to False.
            chunk_size: Chunk size for chunked computation. If None, no chunking.

        Returns:
            Attention output. Shape: [..., Q, c_q].
        """
        q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)

        if attn_bias is not None:
            if len(attn_bias.shape) == len(q.shape):
                assert attn_bias.shape[:-2] == q.shape[:-2]
            else:
                assert len(attn_bias.shape) == len(q.shape) - 1
                assert attn_bias.shape[:-2] == q.shape[:-3]
                attn_bias = attn_bias.unsqueeze(dim=-3)

        if trunked_attn_bias is not None:
            assert n_queries and n_keys
            assert self.local_attention_method == "local_cross_attention"

            if len(trunked_attn_bias.shape) == len(q.shape) + 1:
                assert trunked_attn_bias.shape[:-3] == q.shape[:-2]
            else:
                assert len(trunked_attn_bias.shape) == len(q.shape)
                trunked_attn_bias = trunked_attn_bias.unsqueeze(dim=-4)

        if n_queries and n_keys:
            if self.local_attention_method == "global_attention_with_bias":
                local_attn_bias = create_local_attn_bias(
                    q.shape[-2], n_queries, n_keys, inf=inf, device=q.device
                )
                local_attn_bias = local_attn_bias.reshape(
                    (1,) * (len(q.shape[:-2])) + local_attn_bias.shape
                )
                if attn_bias is not None:
                    if inplace_safe:
                        local_attn_bias += attn_bias
                    else:
                        local_attn_bias = local_attn_bias + attn_bias
                o = _protenix_attention(
                    q=q,
                    k=k,
                    v=v,
                    attn_bias=local_attn_bias,
                    use_efficient_implementation=self.use_efficient_implementation,
                    inplace_safe=inplace_safe,
                )
            elif self.local_attention_method == "local_cross_attention":
                o = _local_attention(
                    q=q,
                    k=k,
                    v=v,
                    n_queries=n_queries,
                    n_keys=n_keys,
                    attn_bias=attn_bias,
                    trunked_attn_bias=trunked_attn_bias,
                    inf=inf,
                    use_efficient_implementation=self.use_efficient_implementation,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            else:
                raise ValueError(
                    f"Invalid local attention method: {self.local_attention_method}"
                )
        else:
            o = _protenix_attention(
                q=q,
                k=k,
                v=v,
                attn_bias=attn_bias,
                use_efficient_implementation=self.use_efficient_implementation,
                inplace_safe=inplace_safe,
            )
        o = o.transpose(-2, -3)
        o = self._wrap_up(o, q_x)
        return o


class ProtenixAttentionPairBias(nn.Module):
    """Attention with pair bias for single representation update.

    Implements Algorithm 24 in AlphaFold3. Updates single representations using
    attention mechanism conditioned on pair representations.
    """

    def __init__(
        self,
        has_s: bool = True,
        create_offset_ln_z: bool = False,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
    ) -> None:
        """Initializes the ProtenixAttentionPairBias module.

        Args:
            has_s: Whether single embedding s is provided. Defaults to True.
            create_offset_ln_z: Whether to create offset for pair LayerNorm.
                Defaults to False.
            n_heads: Number of attention heads. Defaults to 16.
            c_a: Aggregated atom representation dimension. Defaults to 768.
            c_s: Single embedding dimension. Defaults to 384.
            c_z: Pair embedding dimension. Defaults to 128.
            biasinit: Bias initialization value for output projection. Defaults to -2.0.
        """
        super().__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        self.create_offset_ln_z = create_offset_ln_z

        if has_s:
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            self.layernorm_a = ProtenixLayerNorm(c_a)

        self.attention = ProtenixAttention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            zero_init=not self.has_s,
        )
        self.layernorm_z = ProtenixLayerNorm(c_z, create_offset=self.create_offset_ln_z)
        self.linear_nobias_z = ProtenixLinearNoBias(in_features=c_z, out_features=n_heads)

        if self.has_s:
            self.linear_a_last = ProtenixBiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )

    def forward(
        self,
        a: torch.Tensor,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Computes attention with pair bias.

        Args:
            a: Aggregated atom representation. Shape: [..., N_token, c_a].
            s: Single embedding. Shape: [..., N_token, c_s] or None.
            z: Pair embedding. Shape: [..., N_token, N_token, c_z].

        Returns:
            Updated representation. Shape: [..., N_token, c_a] if has_s is False,
                otherwise [..., N_token, c_s].
        """
        if self.has_s:
            a = self.layernorm_a(a, s)
        else:
            a = self.layernorm_a(a)

        # Compute pair bias
        z = self.layernorm_z(z)
        bias = self.linear_nobias_z(z)
        # bias: [..., N_token, N_token, n_heads]
        bias = bias.transpose(-1, -3).transpose(-1, -2)
        # bias: [..., n_heads, N_token, N_token]

        # Attention
        a = self.attention(q_x=a, kv_x=a, attn_bias=bias, inplace_safe=False)

        if self.has_s:
            a = self.linear_a_last(a)

        return a


class ProtenixAttentionPairBiasWithLocalAttn(nn.Module):
    """Attention with pair bias and local attention support.

    Implements Algorithm 24 in AlphaFold3 with support for local attention windows
    to reduce computational complexity for long sequences.
    """

    def __init__(
        self,
        has_s: bool = True,
        create_offset_ln_z: bool = False,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        """Initializes the ProtenixAttentionPairBiasWithLocalAttn module.

        Args:
            has_s: Whether single embedding s is provided. Defaults to True.
            create_offset_ln_z: Whether to create offset for pair LayerNorm.
                Defaults to False.
            n_heads: Number of attention heads. Defaults to 16.
            c_a: Aggregated atom representation dimension. Defaults to 768.
            c_s: Single embedding dimension. Defaults to 384.
            c_z: Pair embedding dimension. Defaults to 128.
            biasinit: Bias initialization value. Defaults to -2.0.
            cross_attention_mode: Whether to use cross-attention with separate
                key/value normalization. Defaults to False.
        """
        super().__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        self.create_offset_ln_z = create_offset_ln_z
        self.cross_attention_mode = cross_attention_mode

        if has_s:
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            if self.cross_attention_mode:
                self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            self.layernorm_a = ProtenixLayerNorm(c_a)
            if self.cross_attention_mode:
                self.layernorm_kv = ProtenixLayerNorm(c_a)

        self.local_attention_method = "local_cross_attention"
        self.attention = ProtenixAttention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            local_attention_method=self.local_attention_method,
            zero_init=not self.has_s,
        )
        self.layernorm_z = ProtenixLayerNorm(c_z, create_offset=self.create_offset_ln_z)
        self.linear_nobias_z = ProtenixLinearNoBias(in_features=c_z, out_features=n_heads)

        if self.has_s:
            self.linear_a_last = ProtenixBiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )

    def local_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(q.shape) + 2

        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [3, 0, 1, 2])

        q = self.attention(
            q_x=q,
            kv_x=kv,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return q

    def standard_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [2, 0, 1])
        q = self.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)
        return q

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
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        if self.cross_attention_mode:
            if self.has_s:
                kv = self.layernorm_kv(a=a, s=s)
            else:
                kv = self.layernorm_kv(a)
        else:
            kv = None

        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                inplace_safe=inplace_safe,
            )

        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a

        return a
