import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from onescience.modules import OneMlp, OneTransformer
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding

class Model(nn.Module):
    """
    Orthogonal Neural Operator (ONO) 模型。
    """
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "ONO"
        self.args = args
        
        # Embedding & Preprocessing 
        if args.unified_pos and args.geotype != "unstructured":
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            dim_x = args.ref ** len(args.shapelist)
            dim_z = args.fun_dim + args.ref ** len(args.shapelist)
        else:
            dim_x = args.fun_dim + args.space_dim
            dim_z = args.fun_dim + args.space_dim

        self.preprocess_x = OneMlp(
            style="StandardMLP",
            input_dim=dim_x,
            output_dim=args.n_hidden,
            hidden_dims=[args.n_hidden * 2],
            activation=args.act,
            use_bias=True
        )
        self.preprocess_z = OneMlp(
            style="StandardMLP",
            input_dim=dim_z,
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

        # ONO Blocks 
        self.blocks = nn.ModuleList([
            OneTransformer(
                style="OrthogonalNeuralBlock", 
                num_heads=args.n_heads,
                hidden_dim=args.n_hidden,
                dropout=args.dropout,
                act=args.act,
                attn_type=args.attn_type,
                mlp_ratio=args.mlp_ratio,
                last_layer=(_ == args.n_layers - 1),
                psi_dim=args.psi_dim,
                out_dim=args.out_dim,
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
            x = torch.cat((x, fx), -1)
            fx = self.preprocess_z(x)
            x = self.preprocess_x(x)
        else:
            fx = self.preprocess_z(x)
            x = self.preprocess_x(x)
            
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden)
            Time_emb = self.time_fc(Time_emb)
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1)
            fx = fx + Time_emb

        for block in self.blocks:
            x, fx = block(x, fx)
        return fx