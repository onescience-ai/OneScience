"""
Protenix Fuser Modules
Implements Pairformer and related fusing layers for Protenix (AlphaFold3)
Reference: Algorithm 17 in AF3
"""
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn

from onescience.models.openfold.dropout import DropoutRowwise
from onescience.models.openfold.primitives import ProtenixLayerNorm
from onescience.models.protenix.modules.primitives import Transition

from onescience.modules.attention.protenixattention import ProtenixAttentionPairBias


class ProtenixPairformerBlock(nn.Module):
    """
    Implements Algorithm 17 [Line2-Line8] in AF3
    A single block of Pairformer that updates pair and single representations.
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        dropout: float = 0.25,
    ) -> None:
        """
        Args:
            n_heads: Number of heads for AttentionPairBias. Defaults to 16.
            c_z: Hidden dim for pair embedding. Defaults to 128.
            c_s: Hidden dim for single embedding. Defaults to 384.
            c_hidden_mul: Hidden dim for triangular multiplication. Defaults to 128.
            c_hidden_pair_att: Hidden dim for triangular attention. Defaults to 32.
            no_heads_pair: Number of heads for triangular attention. Defaults to 4.
            dropout: Dropout ratio. Defaults to 0.25.
        """
        super().__init__()
        self.n_heads = n_heads
        self.c_s = c_s

        # Triangular multiplicative updates (from openfold)
        # These are typically imported from openfold, kept as-is
        from onescience.models.openfold.triangular_multiplicative_update import (
            ProtenixTriangleMultiplicationIncoming,
            ProtenixTriangleMultiplicationOutgoing,
        )
        from onescience.models.openfold.triangular_attention import TriangleAttention

        self.tri_mul_out = ProtenixTriangleMultiplicationOutgoing(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_mul_in = ProtenixTriangleMultiplicationIncoming(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_att_start = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
            bias=False
        )
        self.tri_att_end = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
            bias=False
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.pair_transition = Transition(c_in=c_z, n=4)

        if self.c_s > 0:
            self.attention_pair_bias = ProtenixAttentionPairBias(
                has_s=False, create_offset_ln_z=True, n_heads=n_heads, c_a=c_s, c_z=c_z
            )
            self.single_transition = Transition(c_in=c_s, n=4)

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            s: Single feature [..., N_token, c_s]
            z: Pair embedding [..., N_token, N_token, c_z]
            pair_mask: Pair mask [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient kernel
            use_deepspeed_evo_attention: Whether to use DeepSpeed evo attention
            use_lma: Whether to use low-memory attention
            inplace_safe: Whether inplace operations are safe
            chunk_size: Chunk size for memory-efficient operations

        Returns:
            Updated s and z
        """
        # Triangular multiplicative updates
        if inplace_safe:
            z = self.tri_mul_out(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True
            )
            z = self.tri_mul_in(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True
            )
            z += self.tri_att_start(
                z,
                mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.pair_transition(z)
        else:
            tmu_update = self.tri_mul_out(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=False
            )
            z = z + self.dropout_row(tmu_update)
            del tmu_update

            tmu_update = self.tri_mul_in(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=False
            )
            z = z + self.dropout_row(tmu_update)
            del tmu_update

            z = z + self.dropout_row(
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            z = z.transpose(-2, -3)
            z = z + self.dropout_row(
                self.tri_att_end(
                    z,
                    mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            z = z.transpose(-2, -3)
            z = z + self.pair_transition(z)

        # Single representation update via attention with pair bias
        if self.c_s > 0:
            s = s + self.attention_pair_bias(
                a=s,
                s=None,
                z=z,
            )
            s = s + self.single_transition(s)

        return s, z


class ProtenixPairformerStack(nn.Module):
    """
    Implements Algorithm 17 [PairformerStack] in AF3
    Stack of Pairformer blocks with optional gradient checkpointing.
    """

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_blocks: Number of blocks. Defaults to 48.
            n_heads: Number of heads. Defaults to 16.
            c_z: Hidden dim for pair embedding. Defaults to 128.
            c_s: Hidden dim for single embedding. Defaults to 384.
            dropout: Dropout ratio. Defaults to 0.25.
            blocks_per_ckpt: Number of blocks per activation checkpoint.
                Higher value trades memory for speed. If None, no checkpointing.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            block = ProtenixPairformerBlock(
                n_heads=n_heads, c_z=c_z, c_s=c_s, dropout=dropout
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
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
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: Single feature [..., N_token, c_s]
            z: Pair embedding [..., N_token, N_token, c_z]
            pair_mask: Pair mask [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient kernel
            use_deepspeed_evo_attention: Whether to use DeepSpeed evo attention
            use_lma: Whether to use low-memory attention
            inplace_safe: Whether inplace operations are safe
            chunk_size: Chunk size for memory-efficient operations

        Returns:
            Updated s and z
        """
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False

        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        from onescience.utils.openfold.checkpointing import checkpoint_blocks

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return s, z
