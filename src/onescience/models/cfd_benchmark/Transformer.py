import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import trunc_normal_
from onescience.modules import OneMlp, OneAttention
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
from einops import rearrange, repeat

class Transformer_block(nn.Module):
    """
    Transformer encoder block.
    包含多头自注意力和前馈神经网络，最后一层可选是否直接输出预测维度。
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = OneAttention(
            style="FlashAttention",
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
        )
        
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        self.mlp = OneMlp(
            style="StandardMLP",
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim * mlp_ratio],
            activation=act,
            use_bias=True
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
    """
    标准的 Transformer 模型架构。
    通过堆叠 Transformer_block 处理物理场数据。
    """
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "Transformer"
        self.args = args
        
        ## embedding
        if (
            args.unified_pos and args.geotype != "unstructured"
        ):  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            input_dim = args.fun_dim + args.ref ** len(args.shapelist)
        else:
            input_dim = args.fun_dim + args.space_dim

        self.preprocess = OneMlp(
            style="StandardMLP",
            input_dim=input_dim,
            output_dim=args.n_hidden,
            hidden_dims=[args.n_hidden * 2],
            activation=args.act,
            use_bias=True
        )

        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )

        self.blocks = nn.ModuleList(
            [
                Transformer_block(
                    num_heads=args.n_heads,
                    hidden_dim=args.n_hidden,
                    dropout=args.dropout,
                    act=args.act,
                    mlp_ratio=args.mlp_ratio,
                    out_dim=args.out_dim,
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
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden)
            Time_emb = self.time_fc(Time_emb)
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
            
        return fx