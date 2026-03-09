import torch
import torch.nn as nn
import numpy as np
from timm.layers import trunc_normal_
from onescience.modules import OneMlp, OneTransformer
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding

class Model(nn.Module):
    """
    Transolver 模型。
    通过物理启发的切片机制 (Slicing) 解决 PDE 和物理场预测问题。
    """
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "Transolver"
        self.args = args
        
        ## embedding
        if (
            args.unified_pos and args.geotype != "unstructured"
        ):  
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            self.preprocess = OneMlp(
                style="StandardMLP",
                input_dim=args.fun_dim + args.ref ** len(args.shapelist),
                output_dim=args.n_hidden,
                hidden_dims=[args.n_hidden * 2],
                activation=args.act,
                use_bias=True
            )
        else:
            self.preprocess = OneMlp(
                style="StandardMLP",
                input_dim=args.fun_dim + args.space_dim,
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

        ## models
        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    style="Transolver_block",
                    num_heads=args.n_heads,
                    hidden_dim=args.n_hidden,
                    dropout=args.dropout,
                    act=args.act,
                    mlp_ratio=args.mlp_ratio,
                    out_dim=args.out_dim,
                    slice_num=args.slice_num,
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
            Time_emb = timestep_embedding(T, self.args.n_hidden)
            Time_emb = self.time_fc(Time_emb)
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        return fx

    def unstructured_geo(self, x, fx, T=None):
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

    def forward(self, x, fx, T=None, geo=None):
        if self.args.geotype == "unstructured":
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)