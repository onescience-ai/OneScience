import torch
import torch.nn as nn
import torch_geometric.nn as nng
from onescience.modules import OneMlp

class Model(nn.Module):
    """
    GraphSAGE 模型。

    使用 SAGEConv 进行邻居聚合，并结合 MLP 进行特征编码和解码。
    """
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.__name__ = "GraphSAGE"

        self.nb_hidden_layers = args.n_layers
        self.size_hidden_layers = args.n_hidden
        self.bn_bool = True
        self.activation = nn.ReLU()

        self.encoder = OneMlp(
            style="StandardMLP",
            input_dim=args.fun_dim + args.space_dim,
            output_dim=args.n_hidden,
            hidden_dims=[args.n_hidden * 2],
            activation=args.act,
            use_bias=True
        )
        
        self.decoder = OneMlp(
            style="StandardMLP",
            input_dim=args.n_hidden,
            output_dim=args.out_dim,
            hidden_dims=[args.n_hidden * 2],
            activation=args.act,
            use_bias=True
        )

        # Graph Layers (Keep PyG implementation for consistency)
        self.in_layer = nng.SAGEConv(
            in_channels=args.n_hidden, out_channels=self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(
                nng.SAGEConv(
                    in_channels=self.size_hidden_layers,
                    out_channels=self.size_hidden_layers,
                )
            )

        self.out_layer = nng.SAGEConv(
            in_channels=self.size_hidden_layers, out_channels=self.size_hidden_layers
        )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(
                    nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False)
                )

    def forward(self, x, fx, T=None, geo=None):
        if x.dim() == 3:
            x = x.squeeze(0)  # [1, N, C] → [N, C]
        if fx is not None and fx.dim() == 3:
            fx = fx.squeeze(0)  # [1, N, C] → [N, C]
        if geo.dim() == 3:
            edge_index = geo.squeeze(0)  # [1, 2, E] → [2, E]
        else:
            edge_index = geo

        z = torch.cat((x, fx), dim=-1)
        z = self.encoder(z)
        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)

        z = self.out_layer(z, edge_index)
        z = self.decoder(z)
        return z.unsqueeze(0)  