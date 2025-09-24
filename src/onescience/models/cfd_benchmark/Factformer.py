import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import trunc_normal_
from onescience.models.layers.Basic import MLP, PreNorm, Attention
from onescience.models.layers.Embedding import timestep_embedding, unified_pos_embedding
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PoolingReducer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = PreNorm(in_dim, MLP(hidden_dim, hidden_dim, out_dim))

    def forward(self, x):
        # x: b nx ... c
        x = self.to_in(x)
        ndim = len(x.shape)
        x = x.mean(dim=tuple(range(2, ndim - 1)))
        x = self.out_ffn(x)
        return x  # b nx c


class FactAttnWeight(nn.Module):
    def __init__(self, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = (
            x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        )  # B H N C
        q = self.to_q(x)
        k = self.to_k(x)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        return attn


class FactAttention2D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, shapelist=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.H = shapelist[0]
        self.W = shapelist[1]
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.attn_x = FactAttnWeight(heads, dim_head, dropout)
        self.attn_y = FactAttnWeight(heads, dim_head, dropout)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_x = nn.Sequential(PoolingReducer(inner_dim, inner_dim, inner_dim))
        self.to_y = nn.Sequential(
            Rearrange("b nx ny c -> b ny nx c"),
            PoolingReducer(inner_dim, inner_dim, inner_dim),
        )
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous()
        v = (
            self.to_v(
                x.reshape(B, self.H, self.W, self.heads, self.dim_head).contiguous()
            )
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        res_x = torch.einsum("bhij,bhjmc->bhimc", self.attn_x(self.to_x(x)), v)
        res_y = torch.einsum("bhlm,bhimc->bhilc", self.attn_y(self.to_y(x)), res_x)
        res = rearrange(res_y, "b h i l c -> b (i l) (h c)", h=self.heads)
        return self.to_out(res)


class FactAttention3D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, shapelist=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.H = shapelist[0]
        self.W = shapelist[1]
        self.D = shapelist[2]
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.attn_x = FactAttnWeight(heads, dim_head, dropout)
        self.attn_y = FactAttnWeight(heads, dim_head, dropout)
        self.attn_z = FactAttnWeight(heads, dim_head, dropout)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_x = nn.Sequential(PoolingReducer(inner_dim, inner_dim, inner_dim))
        self.to_y = nn.Sequential(
            Rearrange("b nx ny nz c -> b ny nx nz c"),
            PoolingReducer(inner_dim, inner_dim, inner_dim),
        )
        self.to_z = nn.Sequential(
            Rearrange("b nx ny nz c -> b nz nx ny c"),
            PoolingReducer(inner_dim, inner_dim, inner_dim),
        )
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous()
        v = (
            self.to_v(
                x.reshape(
                    B, self.H, self.W, self.D, self.heads, self.dim_head
                ).contiguous()
            )
            .permute(0, 4, 1, 2, 3, 5)
            .contiguous()
        )

        res_x = torch.einsum("bhij,bhjmsc->bhimsc", self.attn_x(self.to_x(x)), v)
        res_y = torch.einsum("bhlm,bhimsc->bhilsc", self.attn_y(self.to_y(x)), res_x)
        res_z = torch.einsum("bhrs,bhilsc->bhilrc", self.attn_z(self.to_z(x)), res_y)
        res = rearrange(res_z, "b h i l r c -> b (i l r) (h c)", h=self.heads)
        return self.to_out(res)


FACT_ATTENTION = {
    "structured_1D": Attention,
    "structured_2D": FactAttention2D,
    "structured_3D": FactAttention3D,
}


class Factformer_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        geotype="unstructured",
        shapelist=None,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = FACT_ATTENTION[geotype](
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            shapelist=shapelist,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    ## Factformer
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "Factformer"
        self.args = args
        ## embedding
        if args.geotype == "unstructured":
            raise ValueError(
                "Factformer does not support unstructured geometry, please try to integrate GeoFNO layer"
            )
        if args.unified_pos:  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            self.preprocess = MLP(
                args.fun_dim + args.ref ** len(args.shapelist),
                args.n_hidden * 2,
                args.n_hidden,
                n_layers=0,
                res=False,
                act=args.act,
            )
        else:
            self.preprocess = MLP(
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
                Factformer_block(
                    num_heads=args.n_heads,
                    hidden_dim=args.n_hidden,
                    dropout=args.dropout,
                    act=args.act,
                    mlp_ratio=args.mlp_ratio,
                    out_dim=args.out_dim,
                    last_layer=(_ == args.n_layers - 1),
                    geotype=args.geotype,
                    shapelist=args.shapelist,
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

    def structured_geo(self, x, fx, T=None):
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(
                1, x.shape[1], 1
            )
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        return fx

    def unstructured_geo(self, x, fx, T=None):
        raise ValueError(
            "Factformer does not support unstructured geometry, please try to integrate GeoFNO layer"
        )

    def forward(self, x, fx, T=None, geo=None):
        if self.args.geotype == "unstructured":
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)
