# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from onescience.distributed.megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from onescience.distributed.megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from onescience.distributed.megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from onescience.distributed.megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from onescience.distributed.megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from onescience.distributed.megatron.core.ssm.mlp_layer import MLPLayer
from onescience.distributed.megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from onescience.distributed.megatron.core.transformer.enums import AttnMaskType
from onescience.distributed.megatron.core.transformer.mlp import MLP, MLPSubmodules
from onescience.distributed.megatron.core.transformer.spec_utils import ModuleSpec
from onescience.distributed.megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

mamba_stack_spec = ModuleSpec(
    module=MambaStack,
    submodules=MambaStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
    ),
)
