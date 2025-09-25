"""Basic modules for attention and attention embedding.
Contains:
    - AttentionKernel: module for attention operation;
    - AttentionEmbedding: generates query, key and value;
    - HyperAttentionEmbedding: generates hyper-attention kernel;
    - InvariantPointAttentionEmbedding: generates invariant point attention kernel;
"""

import functools
import math
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_uniform, lecun_normal, ones_init, zeros_init
from jax import Array
from jax.experimental.pallas.ops.gpu.attention import mha as _flash_attention
from jax.experimental.pallas.ops.gpu.attention import segment_mask
from ml_collections.config_dict import ConfigDict

from onescience.flax_models.Pt_DiT.common.config import Config

# from ..common.pallas.attention import mha as _flash_attention
from onescience.flax_models.Pt_DiT.common.layers.dense import HyperLoRADense
from onescience.flax_models.Pt_DiT.common.pallas.attention_withbias import (
    mha as _flash_attention_withbias,
)
from onescience.flax_models.Pt_DiT.common.utils import (
    gather_neighbor,
    get_initializer,
    get_rbf,
)

from .transformer import NormBlock


## Adapted from jax.experimental.pallas.ops.attention
@functools.partial(jax.jit, static_argnames=["sm_scale", "causal", "attention_type"])
def _attention(
    q: jnp.ndarray,  # (B, N, H, C)
    k: jnp.ndarray,
    v: jnp.ndarray,
    segment_ids: Optional[jnp.ndarray],
    sm_scale: float = 1.0,
    causal: bool = False,
    attention_type: str = "self",
):
    n_seq_q = q.shape[-3]
    n_seq_k = k.shape[-3]
    logits = jnp.einsum("bqhc,bkhc->bhqk", q, k).astype(jnp.float32)

    mask = None
    if segment_ids is not None:
        if attention_type == "self":
            mask = jnp.expand_dims(segment_mask(segment_ids, segment_ids), 1)
        elif attention_type == "cross":
            mask = jnp.expand_dims(segment_mask(*segment_ids), 1)
        mask = jnp.broadcast_to(mask, logits.shape)
    if causal:
        causal_mask = jnp.tril(jnp.ones((1, 1, n_seq_q, n_seq_k), dtype=bool))
        causal_mask = jnp.broadcast_to(causal_mask, logits.shape)
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
    weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)

    return jnp.einsum("bhqk,bkhc->bqhc", weights, v)


@functools.partial(jax.jit, static_argnames=["sm_scale", "causal", "attention_type"])
def _attention_withbias(
    q: jnp.ndarray,  # (B, N, H, C)
    k: jnp.ndarray,
    v: jnp.ndarray,
    b: jnp.ndarray,  # (B, H, N, N)
    segment_ids: Optional[jnp.ndarray],
    sm_scale: float = 1.0,
    causal: bool = False,
    attention_type: str = "self",
):
    n_seq_q = q.shape[-3]
    n_seq_k = k.shape[-3]
    logits = jnp.einsum("bqhc,bkhc->bhqk", q, k).astype(jnp.float32)

    mask = None
    if segment_ids is not None:
        if attention_type == "self":
            mask = jnp.expand_dims(segment_mask(segment_ids, segment_ids), 1)
        elif attention_type == "cross":
            mask = jnp.expand_dims(segment_mask(*segment_ids), 1)
        mask = jnp.broadcast_to(mask, logits.shape)
    if causal:
        causal_mask = jnp.tril(jnp.ones((1, 1, n_seq_q, n_seq_k), dtype=bool))
        causal_mask = jnp.broadcast_to(causal_mask, logits.shape)
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
    weights = jax.nn.softmax(logits * sm_scale + b.astype(jnp.float32)).astype(q.dtype)

    return jnp.einsum("bhqk,bkhc->bqhc", weights, v)


## adapt from deepmind gemma
_MAX_WAVELENGTH = 10000


def apply_rope(
    inputs: Array,
    positions: Array,
    head_dim: int,
    max_wavelength: int = _MAX_WAVELENGTH,
):
    """apply rope for input.
    Inputs:
        inputs: shape of (B, N, H, C), input;
        positions: shape of (B, N), position index;
        head_dim: head dimension;
        max_wavelength: max wavelength;
    Returns:
        out: shape of (B, N, H, C), output;
    """

    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = max_wavelength**fraction

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class AttentionKernel(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]

    @nn.compact
    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
        b: Array = None,
        m: Array = None,
    ):
        """Attention operation with optional bias and mask.
        Inputs:
            q: shape of (B, Q, H, C), query;
            k: shape of (B, K, H, C), key;
            v: shape of (B, K, H, C), value;
            b: shape of (B, H, Q, K), bias;
            m: shape of (B, Q) or [(B, Q), (B, K)], mask for sequence;
        Returns:
            out: shape of (B, Q, H, C);
        """

        has_bias = self.config.has_bias
        causal_flag = self.config.causal_flag
        flash_attention_flag = self.config.flash_attention_flag
        attention_type = self.config.attention_type
        if flash_attention_flag:
            block_q = self.config.block_q
            block_k = self.config.block_k

        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        batch_size, n_seq, n_head, n_channel = q.shape
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        if flash_attention_flag:
            if has_bias:
                out = _flash_attention_withbias(
                    q,
                    k,
                    v,
                    b,
                    m,
                    sm_scale,
                    causal_flag,
                    attention_type,
                    block_q,
                    block_k,
                )
            else:
                q, k, v = jax.tree_util.tree_map(jnp.float32, (q, k, v))
                out = _flash_attention(
                    q, k, v, m, sm_scale, causal_flag, block_q, block_k
                )
        else:
            if has_bias:
                out = _attention_withbias(
                    q, k, v, b, m, sm_scale, causal_flag, attention_type
                )
            else:
                out = _attention(q, k, v, m, sm_scale, causal_flag, attention_type)

        return out.astype(arr_dtype)


class PostAttention(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]
    hyper_lora_config: Union[Config, ConfigDict] = None

    @nn.compact
    def __call__(self, x: Array, q: Array = None, hyper_var: Array = None):
        """Post attention operation.
        Inputs:
            q: shape of (B, N, H, C), query;
            x: shape of (B, N, H, C), input, usually the output of attention kernel;
        Returns:
            out: shape of (B, N, F), output;
        """

        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        dropout_flag = self.global_config.dropout_flag
        batch_size, n_seq, n_head, n_channel = x.shape
        out_dim = self.config.out_dim
        gating_flag = self.config.gating_flag
        hyper_lora_flag = True if self.hyper_lora_config else False
        if hyper_lora_flag:
            DenseModule = HyperLoRADense
            hyper_var_ = {"hyper_var": hyper_var}
            arg_dict = {
                **self.hyper_lora_config.lora_dense_args,
                "lora_dropout_flag": dropout_flag,
            }
        else:
            DenseModule = nn.Dense
            hyper_var_ = {}
            arg_dict = {}

        x = jnp.reshape(x, (batch_size, n_seq, -1))
        if gating_flag:
            ## (B, N, H*C) -> (B, N, H*C)
            gating_values = DenseModule(
                features=n_channel * n_head,
                use_bias=True,
                dtype=arr_dtype,
                kernel_init=zeros_init(),
                bias_init=ones_init(),
                name="gating",
                **arg_dict,
            )(q.reshape(batch_size, n_seq, -1), **hyper_var_)
            gating_values = jnp.float32(gating_values)
            gating_values = nn.sigmoid(gating_values)
            ## (B, N, H*C) * (B, N, H*C) -> (B, N, H*C)
            gating_values = arr_dtype(gating_values)
            x = x * gating_values

        ## (B, N, H*C) -> (B, N, F)
        x = DenseModule(
            features=out_dim,
            use_bias=True,
            dtype=arr_dtype,
            kernel_init=glorot_uniform(),
            name="output",
            **arg_dict,
        )(x, **hyper_var_)

        return x


class ContinusConvolution(nn.Module):

    module_name: str
    global_config: Union[Config, ConfigDict]

    @nn.compact
    def __call__(self, s_i: Array, s_ij: Array, m_ij: Array, z_ij: Array):
        """apply convolution for single and pair act.
        Input:
            s_i: shape of (B, N, C) or (N, C), single act;
            s_ij: shape of (B, N, n/N, C) or (N, n/N, C), single act for neighbors;
            m_ij: shape of (B, N, n/N), mask for pair act;
            z_ij: shape of (B, N, n/N, C') or (N, n/N, C'), pair act;
        Output:
            s_i_new: shape of (B, N, C) or (N, C), updated single act;
        NOTE:
            this module supports both dense and sparse neighbor mode. while using
            dense mode, the input s_ij could be (B, N, N, C) or (B, 1, N, C).
        """

        arr_type = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        norm_method = self.global_config.norm_method
        norm_small = self.global_config.norm_small

        # (B, N, n, C) -> (B, N, n, C)
        s_ij = nn.Dense(
            features=s_i.shape[-1],
            use_bias=False,
            dtype=arr_type,
            name=f"{self.module_name}_linear_s",
        )(s_ij)
        # (B, N, n, C) - (B, N, 1, C) -> (B, N, n, C)
        s_ij = s_ij - jnp.expand_dims(s_i, axis=-2)
        # (B, N, n, C) * (B, N, n, C) -> (B, N, n, C) -> (B, N, C)
        s_ij = s_ij * nn.Dense(
            features=s_i.shape[-1],
            use_bias=False,
            dtype=arr_type,
            name=f"{self.module_name}_linear_z",
        )(z_ij)
        s_i_new = jnp.sum(s_ij * m_ij[..., None], axis=-2)
        s_i_new = NormBlock(norm_method, norm_small, name=f"{self.module_name}_norm_s")(
            s_i_new
        )

        return s_i_new


class HyperAttentionEmbedding(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]

    @nn.compact
    def __call__(
        self,
        q_i: Array,
        k_i: Array,
        m_j: Array = None,
        z_ij: Array = None,
        m_ij: Array = None,
        neighbor_or_rope_idxs: Array = None,
    ):
        """Modifies q, k and v with hyper-attention kernel.
        Inputs:
            q_i: shape of (B, N, H, C), query;
            k_i: shape of (B, N, H, C), key;
            m_j: shape of (B, N, n) or (B, N, N), mask for neighbor,
            m_ij: shape of (B, N, N) or (B, N, n), mask for pair activation;
            z_ij: shape of (B, N, N, H, C) or (B, N, n, H, C), pair activation;
            neighbor_or_rope_idxs:
                for HAK embedding, neighbor_idxs:
                    shape of (B, N, n), neighbor index for sparse mode, only used for sparse mode;
                for RoPE embedding, rope_indexs:
                    shape of (B, N), position index for RoPE;
        Outputs:
            q_hyp: shape of (B, N, H, C), modified query;
            k_hyp: shape of (B, N, H, C), modified key;
        NOTE:
            This module is only used for self-attention, and so Q/K = N;
        """

        ## basic config
        arr_type = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        sparse_flag = self.global_config.sparse_flag

        kernel_type = self.config.kernel_type
        assert kernel_type in [
            "hak",
            "rope",
        ], "kernel_type must be either 'hak' or 'rope'."

        batch_size, n_seq, n_head, n_channel = q_i.shape
        assert k_i.shape == q_i.shape, "q and k must have the same shape."

        ### Hyper Affine Kernel
        if kernel_type == "hak":

            dim_r = int(self.config.dim_r)

            ## (B, N, H, C) -> (B, H, N, C) -> (BH, N, C)
            q_i = jnp.swapaxes(q_i, -2, -3)
            q_i = jnp.reshape(q_i, (-1, n_seq, n_channel))
            k_i = jnp.swapaxes(k_i, -2, -3)
            k_i = jnp.reshape(k_i, (-1, n_seq, n_channel))

            ## reshape z_ij
            ## (B, N, N/n, H, C) -> (BH, N, n, C)
            z_ij = jnp.transpose(z_ij, (0, 3, 1, 2, 4))
            z_ij = jnp.reshape(z_ij, (batch_size * n_head, n_seq, -1, n_channel))

            if sparse_flag:
                ## gather to get q_j and k_j
                ## (B, N, n) -> (BH, N, n)
                n_i = neighbor_or_rope_idxs[:, None, :, :]
                n_i = jnp.tile(n_i, (1, n_head, 1, 1))
                n_i = jnp.reshape(n_i, (batch_size * n_head, n_seq, -1))
                ## (BH, N, C) -> (BH, N, n, C)
                q_j = gather_neighbor(q_i, n_i, is_pair=False)
                k_j = gather_neighbor(k_i, n_i, is_pair=False)
                ## (B, N, n) * (B, N, n) -> (B, N, n) -> (BH, N, n)
                m_ij = m_ij * m_j
                m_ij = jnp.tile(m_ij[:, None, :, :], (1, n_head, 1, 1))
                m_ij = jnp.reshape(m_ij, (batch_size * n_head, n_seq, -1))

            else:  # dense mode
                ## (BH, N, C) -> (BH, 1, N, C)
                q_j = jnp.expand_dims(q_i, axis=-3)
                k_j = jnp.expand_dims(k_i, axis=-3)
                ## (B, N, N) * (B, N, N) -> (B, N, N) -> (BH, N, N)
                m_ij = m_ij * m_j
                m_ij = jnp.tile(m_ij[:, None, :, :], (1, n_head, 1, 1))
                m_ij = jnp.reshape(m_ij, (batch_size * n_head, n_seq, -1))

            ## get the hyper embedding
            def apply_hak(x, x_ij, y_ij, m_ij, name="x"):

                ## (BH, N, C) -> (BH, N, C)
                x_i = ContinusConvolution(name, self.global_config)(x, x_ij, m_ij, y_ij)
                ## (BH, N, C) -> (BH, N, C*r) -> (BH, N, C, r)
                ## we use zeros init here in left matrix so that the init of kernel is vanilla attention
                dw_x_lora_left = nn.Dense(
                    features=n_channel * dim_r,
                    use_bias=False,
                    kernel_init=zeros_init(),
                    dtype=arr_type,
                    name=f"dw{name}_lora_left",
                )(x_i)
                dw_x_lora_left = jnp.reshape(
                    dw_x_lora_left, (-1, n_seq, n_channel, dim_r)
                )
                dw_x_lora_right = nn.Dense(
                    features=n_channel * dim_r,
                    use_bias=False,
                    dtype=arr_type,
                    name=f"dw{name}_lora_right",
                )(x_i)
                dw_x_lora_right = jnp.reshape(
                    dw_x_lora_right, (-1, n_seq, n_channel, dim_r)
                )
                # (BH, N, C, r) @ (BH, N, C, r) -> (BH, N, C, C)
                dw_x_lora = jnp.einsum(
                    "bqij,bqkj->bqik", dw_x_lora_left, dw_x_lora_right
                )
                # get the diag part with offsets = [-r, -r+1, ..., r-1, r]
                dw_x = (
                    dw_x_lora
                    - jnp.triu(dw_x_lora, dim_r + 1)
                    - jnp.tril(dw_x_lora, -dim_r - 1)
                )
                # (BH, N, C, C) @ (BH, N, C) -> (BH, Q, C)
                x_hyp = x + jnp.einsum("bqij,bqj->bqi", dw_x, x)
                # (BH, N, C) -> (B, N, H, C)
                x_hyp = jnp.reshape(x_hyp, (batch_size, n_head, n_seq, n_channel))
                x_hyp = jnp.swapaxes(x_hyp, -2, -3)

                return x_hyp

            q_hyp = apply_hak(q_i, q_j, z_ij, m_ij, "q")
            k_hyp = apply_hak(k_i, k_j, z_ij, m_ij, "k")

        ### RoPE Kernel
        elif kernel_type == "rope":

            q_hyp = apply_rope(q_i, neighbor_or_rope_idxs, n_channel)
            k_hyp = apply_rope(k_i, neighbor_or_rope_idxs, n_channel)

        return q_hyp, k_hyp


class AttentionEmbedding(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]
    hyper_lora_config: Union[Config, ConfigDict, None] = None

    def setup(self):

        self.arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32

        self.attention_type = self.config.attention_type
        assert self.attention_type in [
            "self",
            "cross",
        ], "attention_type must be either 'self' or 'cross'."
        self.dim_feature = int(self.config.dim_feature)
        self.n_head = int(self.config.n_head)
        assert (
            self.dim_feature % self.n_head == 0
        ), "dim_feature must be divisible by n_head."
        self.n_channel = self.dim_feature // self.n_head
        self.embedding_pair_flag = self.config.embedding_pair_flag

        kernel_initializer = get_initializer(self.config.kernel_initializer)
        arg_dict = {
            "features": self.dim_feature,
            "use_bias": False,
            "dtype": self.arr_dtype,
            "param_dtype": jnp.float32,
            "kernel_init": kernel_initializer(),
        }
        self.hyper_lora_flag = True if self.hyper_lora_config else False
        DenseModule = HyperLoRADense if self.hyper_lora_flag else nn.Dense
        arg_dict = (
            {**arg_dict, **self.hyper_lora_config.lora_dense_args}
            if self.hyper_lora_flag
            else arg_dict
        )
        self.q_gen = DenseModule(name="q_gen", **arg_dict)
        self.k_gen = DenseModule(name="k_gen", **arg_dict)
        self.v_gen = DenseModule(name="v_gen", **arg_dict)
        # self.q_gen = nn.Dense(features=self.dim_feature, use_bias=False, dtype=self.arr_dtype,
        #                       param_dtype=jnp.float32, kernel_init=kernel_initializer(), name='q_gen')
        # self.k_gen = nn.Dense(features=self.dim_feature, use_bias=False, dtype=self.arr_dtype,
        #                       param_dtype=jnp.float32, kernel_init=kernel_initializer(), name='k_gen')
        # self.v_gen = nn.Dense(features=self.dim_feature, use_bias=False, dtype=self.arr_dtype,
        #                       param_dtype=jnp.float32, kernel_init=kernel_initializer(), name='v_gen')
        if self.embedding_pair_flag:
            self.z_gen = DenseModule(name="z_gen", **arg_dict)
            # self.z_gen = nn.Dense(features=self.n_channel*self.n_head, kernel_init=kernel_initializer, name='z_gen',
            #                     use_bias=False, dtype=self.arr_dtype, param_dtype=jnp.float32)

    def __call__(
        self,
        single_act: Union[Tuple, Array],
        pair_act: Array = None,
        hyper_var: Array = None,
    ):
        """Generate query, key and value for attention operation.
        Inputs:
            single_act:
                for self attention:
                    s_i: shape of (B, Q, Fs), single representation;
                for cross attention:
                    sq_i: shape of (B, Q, Fs), single representation;
                    sk_i: shape of (B, K, Fs);
            pair act:
                z_ij: shape of (B, Q, Qn, Fz), pair representation, NOTE: z_ij is only needed for self-attention;
        Outputs:
            q: shape of (B, Q, H, C), query;
            k: shape of (B, K, H, C), key;
            v: shape of (B, K, H, C), value;
            z: shape of (B, Q, Qn, H, C), pair activation;
        """

        hyper_var_ = {"hyper_var": hyper_var} if self.hyper_lora_flag else {}
        if self.attention_type == "self":
            s_i = single_act
            batch_size, n_q, _ = s_i.shape
            n_k = n_q
            q = self.q_gen(s_i, **hyper_var_)  # (B, Q, H*C)
            k = self.k_gen(s_i, **hyper_var_)
            v = self.v_gen(s_i, **hyper_var_)
        else:  # cross
            sq_i, sk_i = single_act
            batch_size, n_q, _ = sq_i.shape
            _, n_k, _ = sk_i.shape
            q = self.q_gen(sq_i, **hyper_var_)  # (B, Q, H*C)
            k = self.k_gen(sk_i, **hyper_var_)  # (B, K, H*C)
            v = self.v_gen(sk_i, **hyper_var_)

        q = jnp.reshape(q, (batch_size, n_q, self.n_head, self.n_channel))
        k = jnp.reshape(k, (batch_size, n_k, self.n_head, self.n_channel))
        v = jnp.reshape(v, (batch_size, n_k, self.n_head, self.n_channel))

        if self.embedding_pair_flag:
            # (B, Q, Qn, Fz) -> (B, Q, Qn, H*C) -> (B, Q, Qn, H, C)
            _, _, n_qneigh, _ = pair_act.shape
            z = self.z_gen(pair_act, **hyper_var_)
            z = jnp.reshape(z, (batch_size, n_q, n_qneigh, self.n_head, self.n_channel))
        else:
            z = None

        return q, k, v, z


class FlashInvariantPointAttention(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]

    def setup(self):

        self.arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        self.sparse_flag = self.global_config.sparse_flag

        self.n_head = int(self.config.n_head)
        self.n_channel = int(self.config.n_channel)
        self.output_channel = int(self.config.output_channel)
        self.n_point_qk = int(self.config.n_point_qk)
        self.n_point_v = int(self.config.n_point_v)

        self.q_point_local = nn.Dense(
            features=self.n_head * 3 * self.n_point_qk,
            kernel_init=lecun_normal(),
            use_bias=False,
            dtype=self.arr_dtype,
            param_dtype=jnp.float32,
        )
        self.kv_point_local = nn.Dense(
            features=self.n_head * 3 * (self.n_point_qk + self.n_point_v),
            kernel_init=lecun_normal(),
            use_bias=False,
            dtype=self.arr_dtype,
            param_dtype=jnp.float32,
        )
        self.project_f = nn.Dense(
            features=self.n_channel,
            kernel_init=lecun_normal(),
            use_bias=True,
            dtype=self.arr_dtype,
            param_dtype=jnp.float32,
        )
        self.project_o = nn.Dense(
            features=self.output_channel,
            kernel_init=lecun_normal(),
            use_bias=True,
            dtype=self.arr_dtype,
            param_dtype=jnp.float32,
        )
        self.rbf = get_rbf(self.config.rbf.module_name)(
            self.config.rbf, self.global_config
        )

        self.scalar_attention_embedding = AttentionEmbedding(
            self.config.attention_embedding, self.global_config
        )
        self.hyper_attention_embedding = HyperAttentionEmbedding(
            self.config.hyper_attention, self.global_config
        )
        self.convolution = ContinusConvolution("s", self.global_config)
        self.scalar_attention = AttentionKernel(
            self.config.scalar_attention, self.global_config
        )
        self.point_attention = AttentionKernel(
            self.config.point_attention, self.global_config
        )

    def apply_to_point(self, point, rotation, translation):
        """Apply rotation and translation to point.
        Inputs:
            point: shape of (B, N, H*P, 3), points;
            rotation: shape of (B, N, 3, 3), rotation matrix;
            translation: shape of (B, N, 3), translation vector;
        Outputs:
            point: shape of (B, N, H*P, 3), transformed points;
        """

        point_out = jnp.einsum("bnmj,bnij->bnmi", point, rotation)
        point_out += translation[:, :, jnp.newaxis, :]

        return point_out

    def __call__(
        self,
        s_i: Array,
        m_i: Array,
        m_j: Array,
        z_ij: Array,
        m_ij: Array,
        r_i: Array,
        t_i: Array,
        n_i: Array = None,
    ):
        """Hyper attention invariant point embedding.
        Inputs:
            s_i: shape of (B, N, Fs), single representation;
            m_i: shape of (B, N), mask for sequence;
            m_j: shape of (B, N, N/n), mask for neighbor;
            z_ij: shape of (B, N, N/n, Fz), pair activation;
            m_ij: shape of (B, N, N/n), mask for pair activation;
            r_i: shape of (B, N, 3, 3), rotation matrix;
            t_i: shape of (B, N, 3), translation vector;
            n_i: shape of (B, N, n), neighbor index;
        Outputs:
            o_i: shape of (B, N, Fs), output;
        """

        batch_size, n_res, _ = s_i.shape
        n_head = self.config.n_head
        n_point_qk = self.config.n_point_qk
        n_point_v = self.config.n_point_v

        ### 0. apply scalar attention embedding
        q_i, k_i, v_i, z_ij = self.scalar_attention_embedding(s_i, z_ij)

        ### 1. create q&k, v points
        ## [B, N, Fs] -> [B, N, H*3*n_point_qk] -> [B, N, H*n_point_qk, 3]
        q_point_local = self.q_point_local(s_i)
        q_point_local = jnp.reshape(
            q_point_local, (batch_size, n_res, n_head * n_point_qk, 3)
        )
        ## project query points into global frame
        q_point_global = self.apply_to_point(q_point_local, r_i, t_i)

        ## [B, N, Fs] -> [B, N, H*3*(n_point_qk + n_point_v)] -> [B, N, H*(n_point_qk + n_point_v), 3]
        kv_point_local = self.kv_point_local(s_i)
        kv_point_local = jnp.reshape(
            kv_point_local, (batch_size, n_res, n_head * (n_point_qk + n_point_v), 3)
        )
        ## project key and value points into global frame
        kv_point_global = self.apply_to_point(kv_point_local, r_i, t_i)
        ## split key and value points: [B, N, H*(n_point_qk + n_point_v), 3] -> [B, N, H*n_point_qk, 3], [B, N, H*n_point_v, 3]
        k_point_global, v_point_global = jnp.split(
            kv_point_global, [n_head * n_point_qk], axis=-2
        )

        ## for q and k point [B, N, H*n_point_qk, 3] -> [B, N, H, n_point_qk, 3] -> [B, H, N, n_point_qk, 3]
        q_point = jnp.reshape(
            q_point_global, (batch_size, n_res, n_head, n_point_qk, 3)
        )
        k_point = jnp.reshape(
            k_point_global, (batch_size, n_res, n_head, n_point_qk, 3)
        )
        q_point = jnp.swapaxes(q_point, -3, -4)
        k_point = jnp.swapaxes(k_point, -3, -4)
        ## for v point [B, N, H*n_point_v, 3] -> [B, N, H, n_point_v, 3]
        v_point = jnp.reshape(v_point_global, (batch_size, n_res, n_head, n_point_v, 3))

        ### 2. create z_ij_ipa
        ## get distance square
        ## [B, H, N, n_point_qk, 3] -> [B*H*n_point_qk, N, 3]
        q_point = jnp.reshape(q_point.swapaxes(-2, -3), (-1, n_res, 3))
        k_point = jnp.reshape(k_point.swapaxes(-2, -3), (-1, n_res, 3))
        if self.sparse_flag:
            n_neighbor = n_i.shape[-1]
            # [B, N, n] -> [B, H*n_point_qk, N, n] -> [..., N, n]
            n_i_tmp = n_i[:, None, :, :]
            n_i_tmp = jnp.tile(n_i_tmp, (1, n_head * n_point_qk, 1, 1))
            n_i_tmp = jnp.reshape(n_i_tmp, (-1, n_res, n_neighbor))
            ## [..., N, 3] -> [..., N, n, 3]
            k_point_j = gather_neighbor(k_point, n_i_tmp, is_pair=False)
            ## [..., N, 3] -> [..., N, 1, 3]
            q_point_i = jnp.expand_dims(q_point, axis=-2)
            ## [..., N, 1, 3] - [..., N, n, 3] -> [..., N, n, 3] -> [..., N, n]
            v_ij = q_point_i - k_point_j
            ## we add a small value here to avoid nan occurs
            d_ij = jnp.linalg.norm(v_ij + 1e-8, axis=-1, keepdims=False)
            ## [..., N, n] -> [..., N, n, n_basis]
            f_ij = self.rbf(d_ij)
            ## [B*H*n_point_qk, N, n, n_basis] -> [BH, n_point_qk, N, n, n_basis] -> [BH, N, n, n_point_qk*n_basis]
            f_ij = jnp.reshape(
                f_ij, (batch_size * n_head, n_point_qk, n_res, n_neighbor, -1)
            )
            f_ij = jnp.transpose(f_ij, (0, 2, 3, 1, 4))
            f_ij = jnp.reshape(f_ij, (batch_size * n_head, n_res, n_neighbor, -1))
            ## [BH, N, n, n_point_qk*n_basis] -> [BH, N, n, C] -> [B, N, n, H, C]
            z_ij_ipa = self.project_f(f_ij)
            z_ij_ipa = jnp.reshape(
                z_ij_ipa, (batch_size, n_head, n_res, n_neighbor, self.n_channel)
            )
            z_ij_ipa = jnp.transpose(z_ij_ipa, (0, 2, 3, 1, 4))
        else:
            ## [..., N, 3] -> [..., 1, N, 3]
            k_point_j = jnp.expand_dims(k_point, axis=-3)
            ## [..., N, 3] -> [..., N, 1, 3]
            q_point_i = jnp.expand_dims(q_point, axis=-2)
            ## [..., N, 1, 3] - [..., 1, N, 3] -> [..., N, N, 3] -> [..., N, N]
            v_ij = q_point_i - k_point_j
            ## we add a small value here to avoid nan occurs
            d_ij = jnp.linalg.norm(v_ij + 1e-8, axis=-1, keepdims=False)
            ## [..., N, N] -> [..., N, N, n_basis]
            f_ij = self.rbf(d_ij)
            ## [B*H*n_point_qk, N, N, n_basis] -> [BH, n_point_qk, N, N, n_basis] -> [BH, N, N, n_point_qk*n_basis]
            f_ij = jnp.reshape(
                f_ij, (batch_size * n_head, n_point_qk, n_res, n_res, -1)
            )
            f_ij = jnp.transpose(f_ij, (0, 2, 3, 1, 4))
            f_ij = jnp.reshape(f_ij, (batch_size * n_head, n_res, n_res, -1))
            ## [BH, N, N, n_point_qk*n_basis] -> [BH, N, N, C] -> [B, N, N, H, C]
            z_ij_ipa = self.project_f(f_ij)
            z_ij_ipa = jnp.reshape(
                z_ij_ipa, (batch_size, n_head, n_res, n_res, self.n_channel)
            )
            z_ij_ipa = jnp.transpose(z_ij_ipa, (0, 2, 3, 1, 4))
        ## [B, N, N/n, H, C] + [B, N, N/n, H, C] -> [B, N, N/n, H, C]
        z_ij = z_ij + z_ij_ipa

        ### 3. go hyperattention embedding
        q_i, k_i = self.hyper_attention_embedding(q_i, k_i, m_j, z_ij, m_ij, n_i)

        ### 4. go attention
        ## continus convolution for z_ij and s_ij: ######### Liyh: convolution for?
        if self.sparse_flag:

            # [B, N, n] -> [B, H, N, n] -> [BH, N, n]
            n_i_tmp = n_i[:, None, :, :]
            n_i_tmp = jnp.tile(n_i_tmp, (1, n_head, 1, 1))
            n_i_tmp = jnp.reshape(n_i_tmp, (batch_size * n_head, n_res, -1))

            # [B, N, H, C] -> [BH, N, C] -> [BH, N, n, C]
            q_j = jnp.swapaxes(q_i, -2, -3).reshape(-1, n_res, self.n_channel)
            q_j = gather_neighbor(q_j, n_i_tmp, False)
            # [B, N, n] * [B, N, n] -> [B, N, n] -> [BH, N, n]
            m_ij_tmp = m_ij * m_j
            m_ij_tmp = jnp.tile(m_ij_tmp[:, None, :, :], (1, n_head, 1, 1))
            m_ij_tmp = jnp.reshape(m_ij_tmp, (batch_size * n_head, n_res, -1))

        else:  # dense mode

            # [B, N, H, C] -> [BH, N, C] -> [BH, 1, N, C]
            q_j = jnp.swapaxes(q_i, -2, -3).reshape(-1, 1, n_res, self.n_channel)
            # [B, N, N] * [B, N, N] -> [B, N, N] -> [BH, N, N]
            m_ij_tmp = m_ij * m_j
            m_ij_tmp = jnp.tile(m_ij_tmp[:, None, :, :], (1, n_head, 1, 1))
            m_ij_tmp = jnp.reshape(m_ij_tmp, (batch_size * n_head, n_res, -1))

        ## [B, N, H, C] -> [BH, N, C]
        q_i_tmp = jnp.swapaxes(q_i, -2, -3).reshape(-1, n_res, self.n_channel)
        ## [B, N, N/n, H, C] -> [BH, N, N/n, C]
        z_ij_tmp = jnp.reshape(
            z_ij.transpose(0, 3, 1, 2, 4),
            (batch_size * n_head, n_res, -1, self.n_channel),
        )
        o_1 = self.convolution(q_i_tmp, q_j, m_ij_tmp, z_ij_tmp)  # [BH, N, C]

        ## flash attention for scalar q&k, v: [B, N, H, C] -> [B, N, H, C]
        o_2 = self.scalar_attention(q_i, k_i, v_i, None, m_i)

        ## flash attention for point v: [B, N, H, n_point_v, 3] -> [B, N, H, n_point_v*3] -> [B, N, H, C] -> [B, N, H, n_point_v*3]
        ## rotation and translation have been applied in line 385
        ## apply padding to 2^n (to be compatible for flash attnetion):
        ## [B, N, H, n_point_v, 3] -> [B, N, H, n_point_v*3] -> [B, N, H, C']
        v_point = jnp.reshape(v_point, (batch_size, n_res, n_head, -1))
        n_channel_points = v_point.shape[-1]
        n_channel_pad = 2 ** math.ceil(math.log2(n_channel_points))
        v_point = jnp.pad(
            v_point,
            ((0, 0), (0, 0), (0, 0), (0, n_channel_pad - n_channel_points)),
            mode="constant",
            constant_values=0.0,
        )
        ## apply point attention: [B, N, H, C'] -> [B, N, H, C'] -> [B, N, H, n_point_v*3] -> [B, N, H*n_point_v, 3]
        o_3 = self.point_attention(q_i, k_i, v_point, None, m_i)
        o_3, _ = jnp.split(
            o_3,
            [
                n_channel_points,
            ],
            axis=-1,
        )
        o_3 = jnp.reshape(o_3, (batch_size, n_res, -1, 3))
        ## apply reverse translation and rotation: [B, N, H*n_point_v, 3] -> [B, N, H*n_point_v, 3]
        ## o_3 = r_i^T @ o_3 - t_i
        o_3 = self.apply_to_point(o_3, jnp.transpose(r_i, (0, 1, 3, 2)), -t_i)
        ## calculate vector norm: [B, N, H*n_point_v, 3] -> [B, N, H*n_point_v]
        o_3_norm = jnp.linalg.norm(o_3, axis=-1, keepdims=False)

        ## 4. output projection
        ## [BH, N, C] -> [B, N, H, C] -> [B, N, H*C]
        o_1 = jnp.reshape(o_1, (batch_size, n_head, n_res, self.n_channel))
        o_1 = jnp.swapaxes(o_1, -2, -3).reshape(batch_size, n_res, -1)
        o_2 = jnp.reshape(o_2, (batch_size, n_res, -1))  # [B, N, H*C]
        o_3 = jnp.reshape(o_3, (batch_size, n_res, -1))  # [B, N, H*n_point_v*3]
        o = jnp.concatenate((o_1, o_2, o_3, o_3_norm), axis=-1)  # [B, N, ...]
        o = self.project_o(o)  # [B, N, Cout]

        return o
