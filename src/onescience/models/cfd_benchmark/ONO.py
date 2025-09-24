import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import trunc_normal_
from onescience.models.layers.Basic import (
    MLP,
    LinearAttention,
    FlashAttention,
    SelfAttention as LinearSelfAttention,
)
from onescience.models.layers.Embedding import timestep_embedding, unified_pos_embedding
from einops import rearrange, repeat
import warnings


def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        if torch.isnan(L).any():
            raise RuntimeError
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise ValueError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10**i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e


class ONOBlock(nn.Module):
    """ONO encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        attn_type="nystrom",
        mlp_ratio=4,
        last_layer=False,
        momentum=0.9,
        psi_dim=8,
        out_dim=1,
    ):
        super().__init__()
        self.momentum = momentum
        self.psi_dim = psi_dim

        self.register_buffer("feature_cov", torch.zeros(psi_dim, psi_dim))
        self.register_parameter("mu", nn.Parameter(torch.zeros(psi_dim)))
        self.ln_1 = nn.LayerNorm(hidden_dim)
        if attn_type == "nystrom":
            from nystrom_attention import NystromAttention

            self.Attn = NystromAttention(
                hidden_dim,
                heads=num_heads,
                dim_head=hidden_dim // num_heads,
                dropout=dropout,
            )
        elif attn_type == "linear":
            self.Attn = LinearAttention(
                hidden_dim,
                heads=num_heads,
                dim_head=hidden_dim // num_heads,
                dropout=dropout,
                attn_type="galerkin",
            )
        elif attn_type == "selfAttention":
            self.Attn = LinearSelfAttention(
                hidden_dim,
                heads=num_heads,
                dim_head=hidden_dim // num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError("Attn type only supports nystrom or linear")
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        self.proj = nn.Linear(hidden_dim, psi_dim)
        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp2 = (
            nn.Linear(hidden_dim, out_dim)
            if last_layer
            else MLP(
                hidden_dim,
                hidden_dim * mlp_ratio,
                hidden_dim,
                n_layers=0,
                res=False,
                act=act,
            )
        )

    def forward(self, x, fx):
        x = self.Attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        x_ = self.proj(x)
        if self.training:
            batch_cov = torch.einsum("blc, bld->cd", x_, x_) / x_.shape[0] / x_.shape[1]
            with torch.no_grad():
                self.feature_cov.mul_(self.momentum).add_(
                    batch_cov, alpha=1 - self.momentum
                )
        else:
            batch_cov = self.feature_cov
        L = psd_safe_cholesky(batch_cov)
        L_inv_T = L.inverse().transpose(-2, -1)
        x_ = x_ @ L_inv_T

        fx = (x_ * torch.nn.functional.softplus(self.mu)) @ (
            x_.transpose(-2, -1) @ fx
        ) + fx
        fx = self.mlp2(self.ln_3(fx))

        return x, fx


class Model(nn.Module):
    ## speed up with flash attention
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "ONO"
        self.args = args
        ## embedding
        if (
            args.unified_pos and args.geotype != "unstructured"
        ):  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            self.preprocess_x = MLP(
                args.ref ** len(args.shapelist),
                args.n_hidden * 2,
                args.n_hidden,
                n_layers=0,
                res=False,
                act=args.act,
            )
            self.preprocess_z = MLP(
                args.fun_dim + args.ref ** len(args.shapelist),
                args.n_hidden * 2,
                args.n_hidden,
                n_layers=0,
                res=False,
                act=args.act,
            )
        else:
            self.preprocess_x = MLP(
                args.fun_dim + args.space_dim,
                args.n_hidden * 2,
                args.n_hidden,
                n_layers=0,
                res=False,
                act=args.act,
            )
            self.preprocess_z = MLP(
                args.fun_dim + args.space_dim,
                args.n_hidden * 2,
                args.n_hidden,
                n_layers=0,
                res=False,
                act=args.act,
            )
        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )

        ## models
        self.blocks = nn.ModuleList(
            [
                ONOBlock(
                    num_heads=args.n_heads,
                    hidden_dim=args.n_hidden,
                    dropout=args.dropout,
                    act=args.act,
                    mlp_ratio=args.mlp_ratio,
                    out_dim=args.out_dim,
                    psi_dim=args.psi_dim,
                    attn_type=args.attn_type,
                    last_layer=(_ == args.n_layers - 1),
                )
                for _ in range(args.n_layers)
            ]
        )
        self.placeholder = nn.Parameter(
            (1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float)
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None, geo=None):
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            x = torch.cat((x, fx), -1)
            fx = self.preprocess_z(x)
            x = self.preprocess_x(x)
        else:
            fx = self.preprocess_z(x)
            x = self.preprocess_x(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(
                1, x.shape[1], 1
            )
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            x, fx = block(x, fx)
        return fx
