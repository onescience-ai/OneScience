import torch
import torch.nn as nn
from onescience.models.layers.Basic import MLP
from onescience.models.layers.Embedding import timestep_embedding, unified_pos_embedding


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "DeepONet"
        self.args = args

        # 位置处理
        if args.unified_pos and args.geotype != "unstructured":
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            trunk_in_dim = args.ref ** len(args.shapelist)
        else:
            trunk_in_dim = args.space_dim

        # 分支网络输入维度（函数输入+时间嵌入）
        branch_in_dim = args.fun_dim
        if args.time_input:
            branch_in_dim += args.n_hidden  # 增加时间嵌入维度

        # 分支网络（函数空间）
        self.branch_net = MLP(
            branch_in_dim,
            args.n_hidden,
            args.n_hidden,
            n_layers=args.branch_depth,
            res=False,
            act=args.act,
        )

        # 主干网络（物理空间）
        self.trunk_net = MLP(
            trunk_in_dim,
            args.n_hidden,
            args.n_hidden,
            n_layers=args.trunk_depth,
            res=False,
            act=args.act,
        )

        # 时间处理层
        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )

        # 输出层 - 修改为支持多个输出通道
        self.out_layer = nn.Linear(args.n_hidden, args.out_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, args.out_dim))

    def structured_forward(self, x, fx, T=None, geo=None):
        B, N, _ = x.shape  # x: [B, N, space_dim]
        if self.args.unified_pos:
            x = self.pos.repeat(B, 1, 1)  # [B, N, d]

        # 处理时间信息
        if T is not None and self.args.time_input:
            # 确保时间输入维度正确 [B, 1] -> [B]
            T = T.view(-1)

            # 生成时间嵌入 [B, D]
            T_emb = timestep_embedding(T, self.args.n_hidden)

            # 通过时间处理层 [B, D] -> [B, D]
            T_emb = self.time_fc(T_emb)

            # 扩展时间嵌入以匹配空间点 [B, D] -> [B, N, D]
            T_emb_expanded = T_emb.unsqueeze(1).expand(-1, N, -1)

            # 将时间嵌入连接到函数输入
            if fx is not None:
                # [B, N, fun_dim] + [B, N, D] = [B, N, fun_dim + D]
                fx = torch.cat([fx, T_emb_expanded], dim=-1)
            else:
                # 如果没有函数输入，使用时间嵌入作为函数输入
                fx = T_emb_expanded  # [B, N, D]

        # 分支网络处理
        branch_feat = self.branch_net(fx)  # [B, N, D]

        # 主干网络处理
        trunk_feat = self.trunk_net(x)  # [B, N, D]
        # 点积操作 + 输出层
        # 点积结果: [B, N, 1]
        inner = branch_feat * trunk_feat
        out = self.out_layer(inner) + self.bias
        # out = self.out_layer(inner)
        # 应用输出层: [B, N, 1] -> [B, N, out_dim]
        return out

    def unstructured_forward(self, x, fx, T=None):
        B, N, _ = x.shape
        if T is not None and self.args.time_input:
            T = T.view(-1)
            T_emb = timestep_embedding(T, self.args.n_hidden)
            T_emb = self.time_fc(T_emb)
            T_emb_expanded = T_emb.unsqueeze(1).expand(-1, N, -1)
            if fx is not None:
                fx = torch.cat([fx, T_emb_expanded], dim=-1)
            else:
                fx = T_emb_expanded

        branch_feat = self.branch_net(fx)
        trunk_feat = self.trunk_net(x)

        inner = branch_feat * trunk_feat
        out = self.out_layer(inner) + self.bias
        return out

    def forward(self, x, fx, T=None, geo=None):
        if self.args.geotype == "unstructured":
            return self.unstructured_forward(x, fx, T)
        else:
            return self.structured_forward(x, fx, T)
