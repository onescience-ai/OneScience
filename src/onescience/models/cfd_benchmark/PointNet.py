import torch
import torch.nn as nn
import torch_geometric.nn as nng
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
from onescience.modules import OneMlp

class Model(nn.Module):
    """
    PointNet 模型。
    
    用于处理点云数据，通过 MLP 提取局部特征，并使用全局最大池化提取全局特征。
    """
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "PointNet"

        # 1. Input Block
        self.in_block = OneMlp(
            style="StandardMLP",
            input_dim=args.n_hidden,
            output_dim=args.n_hidden * 2,
            hidden_dims=[args.n_hidden * 2],
            activation=args.act,
            use_bias=True
        )

        # 2. Max Pooling Block
        self.max_block = OneMlp(
            style="StandardMLP",
            input_dim=args.n_hidden * 2,
            output_dim=args.n_hidden * 32,
            hidden_dims=[args.n_hidden * 8],
            activation=args.act,
            use_bias=True
        )

        # 3. Output Block
        self.out_block = OneMlp(
            style="StandardMLP",
            input_dim=args.n_hidden * (2 + 32), # 34 * hidden
            output_dim=args.n_hidden * 4,
            hidden_dims=[args.n_hidden * 16],
            activation=args.act,
            use_bias=True
        )

        # 4. Encoder
        self.encoder = OneMlp(
            style="StandardMLP",
            input_dim=args.fun_dim + args.space_dim,
            output_dim=args.n_hidden,
            hidden_dims=[args.n_hidden * 2],
            activation=args.act,
            use_bias=True
        )

        # 5. Decoder
        self.decoder = OneMlp(
            style="StandardMLP",
            input_dim=args.n_hidden,
            output_dim=args.out_dim,
            hidden_dims=[args.n_hidden * 2],
            activation=args.act,
            use_bias=True
        )

        self.fcfinal = nn.Linear(args.n_hidden * 4, args.n_hidden)

    def forward(self, x, fx, T=None, geo=None):
        if geo is None:
            raise ValueError("Please provide edge index for Graph Neural Networks")
        
        # 兼容 batch_size = 1 输入
        if x.dim() == 3:
            x = x.squeeze(0)  # [1, N, C] → [N, C]
        if fx is not None and fx.dim() == 3:
            fx = fx.squeeze(0)

        assert (
            x.size(0) > 0 # Simple check
        ), "Input cannot be empty"

        # 构造 batch 索引
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # 编码 + 局部特征提取
        z = torch.cat((x, fx), dim=-1).float()
        z = self.encoder(z)
        z = self.in_block(z)

        # 全局特征（max pooling）
        global_coef = self.max_block(z)
        global_coef = nng.global_max_pool(global_coef, batch=batch)

        # 重复 global coef 到每个点
        nb_points = torch.tensor([batch.shape[0]], device=z.device)
        global_coef = global_coef.repeat_interleave(nb_points, dim=0)

        # 拼接全局 + 局部特征
        z = torch.cat([z, global_coef], dim=1)
        z = self.out_block(z)
        z = self.fcfinal(z)
        z = self.decoder(z)

        return z.unsqueeze(0) 