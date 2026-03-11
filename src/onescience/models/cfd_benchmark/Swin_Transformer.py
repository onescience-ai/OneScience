import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import trunc_normal_
from onescience.modules import OneMlp, OneTransformer
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding

class BasicLayer(nn.Module):
    """
    Swin Transformer Layer (Stage).
    封装了多个 SwinTransformerBlock。
    """
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            OneTransformer(
                style="SwinTransformerBlock",
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                norm_layer=norm_layer,
                fused_window_process=fused_window_process,
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Model(nn.Module):
    """
    Swin Transformer 主模型。
    """
    def __init__(self, args, device, window_size=4):
        super(Model, self).__init__()
        self.__name__ = "SwinTransformer"
        self.args = args
        
        if args.geotype != "structured_2D":
            raise ValueError("Swin Transformer only supports Structured 2D geometry")

        # 1. Embedding & Preprocessing
        if args.unified_pos: 
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            in_dim = args.fun_dim + args.ref ** len(args.shapelist)
        else:
            in_dim = args.fun_dim + args.space_dim

        self.preprocess = OneMlp(
            style="StandardMLP",
            input_dim=in_dim,
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
            
        self.placeholder = nn.Parameter(
            (1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float)
        )
        
        self.padding = [
            (window_size - size % window_size) % window_size for size in args.shapelist
        ]
        self.augmented_resolution = [
            (self.padding[i] + args.shapelist[i]) for i in range(len(self.padding))
        ]
        
        # 3. Swin Layers
        self.blocks = nn.ModuleList([
            BasicLayer(
                dim=args.n_hidden,
                input_resolution=self.augmented_resolution,
                depth=2,
                num_heads=args.n_heads,
                window_size=window_size,
            )
            for _ in range(args.n_layers)
        ])
        
        # 4. Projectors
        self.fc1 = nn.Linear(args.n_hidden, args.n_hidden * 2)
        self.fc2 = nn.Linear(args.n_hidden * 2, args.out_dim)
        
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
        B, N, _ = x.shape
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
            
        ## aug shape
        fx = fx.permute(0, 2, 1).reshape(B, self.args.n_hidden, *self.args.shapelist)
        
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 2:
                fx = F.pad(fx, [0, self.padding[1], 0, self.padding[0]])
            elif len(self.args.shapelist) == 3:
                fx = F.pad(fx, [0, self.padding[2], 0, self.padding[1], 0, self.padding[0]])
                
        fx = fx.reshape(B, self.args.n_hidden, -1).permute(0, 2, 1)
        
        ## swin transformer
        for block in self.blocks:
            fx = block(fx)
            
        ## back to original shape
        fx = fx.permute(0, 2, 1).reshape(
            B, self.args.n_hidden, *self.augmented_resolution
        )
        
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 2:
                fx = fx[..., : -self.padding[0], : -self.padding[1]]
            elif len(self.args.shapelist) == 3:
                fx = fx[..., : -self.padding[0], : -self.padding[1], : -self.padding[2]]
                
        fx = fx.reshape(B, self.args.n_hidden, -1).permute(0, 2, 1)
        
        ## projection
        fx = self.fc1(fx)
        fx = F.gelu(fx)
        fx = self.fc2(fx)
        return fx

    def forward(self, x, fx, T=None, geo=None):
        if self.args.geotype == "structured_2D":
            return self.structured_geo(x, fx, T)
        else:
            raise ValueError("Swin Transformer only supports Structured 2D geometry")