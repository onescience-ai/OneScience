import torch
import torch.nn as nn
import torch_geometric.nn as nng
from onescience.models.layers.Embedding import unified_pos_embedding
from onescience.models.layers.Basic import MLP


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "PointNet"

        self.in_block = MLP(
            args.n_hidden,
            args.n_hidden * 2,
            args.n_hidden * 2,
            n_layers=0,
            res=False,
            act=args.act,
        )
        self.max_block = MLP(
            args.n_hidden * 2,
            args.n_hidden * 8,
            args.n_hidden * 32,
            n_layers=0,
            res=False,
            act=args.act,
        )

        self.out_block = MLP(
            args.n_hidden * (2 + 32),
            args.n_hidden * 16,
            args.n_hidden * 4,
            n_layers=0,
            res=False,
            act=args.act,
        )

        self.encoder = MLP(
            args.fun_dim + args.space_dim,
            args.n_hidden * 2,
            args.n_hidden,
            n_layers=0,
            res=False,
            act=args.act,
        )
        self.decoder = MLP(
            args.n_hidden,
            args.n_hidden * 2,
            args.out_dim,
            n_layers=0,
            res=False,
            act=args.act,
        )

        self.fcfinal = nn.Linear(args.n_hidden * 4, args.n_hidden)

    def forward(self, x, fx, T=None, geo=None):
        if geo is None:
            raise ValueError("Please provide edge index for Graph Neural Networks")
        assert (
            x.size(0) == 1
        ), "This model only supports batch_size=1. Please modify code for general batching."

        # 兼容 batch_size = 1 输入：去除 batch 维度
        if x.dim() == 3:
            x = x.squeeze(0)  # [1, N, C] → [N, C]
        if fx is not None and fx.dim() == 3:
            fx = fx.squeeze(0)

        # 构造 batch 索引（所有点属于 batch 0）
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

        return z.unsqueeze(0)  # 输出 shape: [1, N, C]
