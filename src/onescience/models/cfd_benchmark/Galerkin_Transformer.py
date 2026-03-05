# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from timm.layers import trunc_normal_
# from onescience.modules.layer.layers.Basic import MLP
# from onescience.modules.attention.linearattention import LinearAttention
# from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange


# class Galerkin_Transformer_block(nn.Module):
#     """Transformer encoder block."""

#     def __init__(
#         self,
#         num_heads: int,
#         hidden_dim: int,
#         dropout: float,
#         act="gelu",
#         mlp_ratio=4,
#         last_layer=False,
#         out_dim=1,
#     ):
#         super().__init__()
#         self.last_layer = last_layer
#         self.ln_1 = nn.LayerNorm(hidden_dim)
#         self.ln_1a = nn.LayerNorm(hidden_dim)
#         self.Attn = LinearAttention(
#             hidden_dim,
#             heads=num_heads,
#             dim_head=hidden_dim // num_heads,
#             dropout=dropout,
#             attn_type="galerkin",
#         )
#         self.ln_2 = nn.LayerNorm(hidden_dim)
#         self.mlp = MLP(
#             hidden_dim,
#             hidden_dim * mlp_ratio,
#             hidden_dim,
#             n_layers=0,
#             res=False,
#             act=act,
#         )
#         if self.last_layer:
#             self.ln_3 = nn.LayerNorm(hidden_dim)
#             self.mlp2 = nn.Linear(hidden_dim, out_dim)

#     def forward(self, fx):
#         fx = self.Attn(self.ln_1(fx), self.ln_1a(fx)) + fx
#         fx = self.mlp(self.ln_2(fx)) + fx
#         if self.last_layer:
#             return self.mlp2(self.ln_3(fx))
#         else:
#             return fx


# class Model(nn.Module):
#     ## Galerkin_Transformer
#     def __init__(self, args, device):
#         super(Model, self).__init__()
#         self.__name__ = "Galerkin_Transformer"
#         self.args = args
#         ## embedding
#         if (
#             args.unified_pos and args.geotype != "unstructured"
#         ):  # only for structured mesh
#             self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
#             self.preprocess = MLP(
#                 args.fun_dim + args.ref ** len(args.shapelist),
#                 args.n_hidden * 2,
#                 args.n_hidden,
#                 n_layers=0,
#                 res=False,
#                 act=args.act,
#             )
#         else:
#             self.preprocess = MLP(
#                 args.fun_dim + args.space_dim,
#                 args.n_hidden * 2,
#                 args.n_hidden,
#                 n_layers=0,
#                 res=False,
#                 act=args.act,
#             )
#         if args.time_input:
#             self.time_fc = nn.Sequential(
#                 nn.Linear(args.n_hidden, args.n_hidden),
#                 nn.SiLU(),
#                 nn.Linear(args.n_hidden, args.n_hidden),
#             )

#         ## models
#         self.blocks = nn.ModuleList(
#             [
#                 Galerkin_Transformer_block(
#                     num_heads=args.n_heads,
#                     hidden_dim=args.n_hidden,
#                     dropout=args.dropout,
#                     act=args.act,
#                     mlp_ratio=args.mlp_ratio,
#                     out_dim=args.out_dim,
#                     last_layer=(_ == args.n_layers - 1),
#                 )
#                 for _ in range(args.n_layers)
#             ]
#         )
#         self.placeholder = nn.Parameter(
#             (1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float)
#         )
#         self.initialize_weights()

#     def initialize_weights(self):
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x, fx, T=None, geo=None):
#         if self.args.unified_pos:
#             x = self.pos.repeat(x.shape[0], 1, 1)
#         if fx is not None:
#             fx = torch.cat((x, fx), -1)
#             fx = self.preprocess(fx)
#         else:
#             fx = self.preprocess(x)
#         fx = fx + self.placeholder[None, None, :]

#         if T is not None:    
#             Time_emb = timestep_embedding(T, self.args.n_hidden) # (B, C)
#             Time_emb = self.time_fc(Time_emb) # (B, C)
            
#             if Time_emb.ndim == 2:
#                 Time_emb = Time_emb.unsqueeze(1) # (B, 1, C)
                
#             fx = fx + Time_emb # Broadcasting: (B, N, C) + (B, 1, C) -> (B, N, C)

#         for block in self.blocks:
#             fx = block(fx)
#         return fx



import torch
import torch.nn as nn
from timm.layers import trunc_normal_

# --- 引入模块工厂 ---
from onescience.modules import OneMlp, OneTransformer
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding

class Model(nn.Module):
    """
    Galerkin Transformer 模型。
    
    使用 Galerkin 线性注意力机制处理物理场数据的 Transformer 架构。
    """
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "Galerkin_Transformer"
        self.args = args
        
        # 1. Embedding & Preprocessing (使用 OneMlp)
        # -----------------------------------------------------------
        input_dim = args.fun_dim
        if args.unified_pos and args.geotype != "unstructured":
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            input_dim += args.ref ** len(args.shapelist)
        else:
            input_dim += args.space_dim

        # 使用 OneMlp (StandardMLP) 替换 Basic.MLP
        # 对应原代码: MLP(input_dim, hidden*2, hidden, n_layers=0)
        self.preprocess = OneMlp(
            style="StandardMLP",
            input_dim=input_dim,
            output_dim=args.n_hidden,
            hidden_dims=[args.n_hidden * 2], # 中间层
            activation=args.act,
            use_bias=True
        )

        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )

        # 2. Transformer Blocks (使用 OneTransformer)
        # -----------------------------------------------------------
        # 使用工厂实例化 Galerkin_Transformer_block
        self.blocks = nn.ModuleList([
            OneTransformer(
                style="Galerkin_Transformer_block",
                num_heads=args.n_heads,
                hidden_dim=args.n_hidden,
                dropout=args.dropout,
                act=args.act,
                mlp_ratio=args.mlp_ratio,
                out_dim=args.out_dim,
                last_layer=(_ == args.n_layers - 1)
            )
            for _ in range(args.n_layers)
        ])
        
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
            Time_emb = timestep_embedding(T, self.args.n_hidden) # (B, C)
            Time_emb = self.time_fc(Time_emb) # (B, C)
            
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1) # (B, 1, C)
                
            fx = fx + Time_emb # Broadcasting: (B, N, C) + (B, 1, C) -> (B, N, C)

        for block in self.blocks:
            fx = block(fx)
        return fx