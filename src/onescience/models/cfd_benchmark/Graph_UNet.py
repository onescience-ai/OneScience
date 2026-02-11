import torch
import torch.nn as nn
import torch_geometric.nn as nng
import random
from onescience.models.layers.Basic import MLP
from onescience.modules.resample.SpatialGraphDownsample import SpatialGraphDownsample
from onescience.modules.resample.SpatialGraphUpsample import SpatialGraphUpsample

class Model(nn.Module):
    def __init__(
        self,
        args,
        device,
        pool="random",
        scale=5,
        list_r=[0.05, 0.2, 0.5, 1, 10],
        pool_ratio=[0.5, 0.5, 0.5, 0.5, 0.5],
        max_neighbors=64,
        layer="SAGE",
        head=2,
    ):
        super(Model, self).__init__()
        self.__name__ = "GUNet"
        
        # 参数绑定
        self.L = scale
        self.layer = layer
        self.pool_type = pool
        self.pool_ratio = pool_ratio
        self.list_r = list_r
        self.size_hidden = args.n_hidden
        self.dim_enc = args.n_hidden
        self.bn_bool = True
        self.res = False
        self.head = head
        self.activation = nn.ReLU()
        
        # MLP Encoder / Decoder
        self.encoder = MLP(
            args.fun_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False, act=args.act
        )
        self.decoder = MLP(
            args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act
        )

        # Down Path Layers
        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.down_bns = nn.ModuleList()

        # Level 0 (Initial)
        self._add_conv_layer(self.down_convs, self.dim_enc, self.size_hidden)
        if self.bn_bool:
            self._add_bn_layer(self.down_bns, self.size_hidden)

        # Level 1 to L-1
        current_dim = self.size_hidden
        for n in range(self.L - 1):
            # Downsample Module
            self.down_samples.append(
                SpatialGraphDownsample(
                    in_channels=current_dim,
                    ratio=self.pool_ratio[n],
                    r=self.list_r[n],
                    max_num_neighbors=max_neighbors,
                    pool_method=self.pool_type
                )
            )
            
            # Conv Layer
            in_c = current_dim
            out_c = 2 * current_dim if layer == "SAGE" else current_dim
            self._add_conv_layer(self.down_convs, in_c, out_c)
            current_dim = out_c
            
            if self.bn_bool:
                self._add_bn_layer(self.down_bns, current_dim)

        # Up Path Layers
        self.up_convs = nn.ModuleList()
        self.up_sampler = SpatialGraphUpsample()
        self.up_bns = nn.ModuleList()
        
        curr_h_init = args.n_hidden
        
        # Up Layer 0 (Top Layer)
        if self.layer == "SAGE":
            self.up_convs.append(nng.SAGEConv(3 * curr_h_init, self.dim_enc))
            curr_h_init = 2 * curr_h_init
        elif self.layer == "GAT":
            self.up_convs.append(nng.GATConv(2 * self.head * curr_h_init, self.dim_enc, heads=2, concat=False))
        
        if self.bn_bool:
             self.up_bns.append(nng.BatchNorm(self.dim_enc, track_running_stats=False))

        # Up Layer 1 to L-1 (Middle Layers)
        for n in range(1, self.L - 1):
            if self.layer == "SAGE":
                self.up_convs.append(nng.SAGEConv(3 * curr_h_init, curr_h_init))
                bn_dim = curr_h_init
                curr_h_init = 2 * curr_h_init
            elif self.layer == "GAT":
                self.up_convs.append(nng.GATConv(2 * self.head * curr_h_init, curr_h_init, heads=2, concat=True))
                bn_dim = curr_h_init * 2 # GAT concat=True
            
            if self.bn_bool:
                self.up_bns.append(nng.BatchNorm(bn_dim, track_running_stats=False))

    def _add_conv_layer(self, module_list, in_c, out_c):
        if self.layer == "SAGE":
            module_list.append(nng.SAGEConv(in_c, out_c))
        elif self.layer == "GAT":
            module_list.append(nng.GATConv(in_c, out_c, heads=self.head, concat=True, add_self_loops=False))

    def _add_bn_layer(self, module_list, in_c):
        dim = in_c * self.head if self.layer == "GAT" else in_c
        module_list.append(nng.BatchNorm(dim, track_running_stats=False))

    def forward(self, x, fx, T=None, geo=None):.
        if geo is None: raise ValueError("Edge index required")
        if fx.dim() == 3: fx = fx.squeeze(0)
        if geo.dim() == 3: edge_index = geo.squeeze(0)
        else: edge_index = geo
        
        x = fx 
        
        # Encoder
        z = self.encoder(x)
        if self.res: z_res = z.clone()

        # Downsampling Path
        skip_connections = [] 
        pos_history = []     
        edge_index_history = [edge_index.clone()]
        
        # Level 0 Conv
        z = self.down_convs[0](z, edge_index)
        if self.bn_bool: z = self.down_bns[0](z)
        z = self.activation(z)
        
        skip_connections.append(z.clone())
        current_pos = x[:, :2] 
        pos_history.append(current_pos.clone())

        # Levels 1 to L-1
        for n in range(self.L - 1):
            # A. 下采样
            z, current_pos, edge_index, _ = self.down_samples[n](z, current_pos, edge_index)
            
            pos_history.append(current_pos.clone())
            edge_index_history.append(edge_index.clone()) # 记录新图结构

            # B. 卷积
            z = self.down_convs[n+1](z, edge_index)
            if self.bn_bool: z = self.down_bns[n+1](z)
            z = self.activation(z)
            
            skip_connections.append(z.clone())
        
        for n in range(self.L - 1, 0, -1):
            layer_idx = n - 1 # 对应 up_convs 的索引
            
            # 获取数据
            pos_low = pos_history[n]
            pos_high = pos_history[n-1]
            z_skip = skip_connections[n-1]
            
            target_edge_index = edge_index_history[n-1]

            # 上采样
            z = self.up_sampler(z, pos_low, pos_high)
            
            # 拼接
            z = torch.cat([z, z_skip], dim=1)
            
            # 卷积 (使用正确的 edge_index)
            z = self.up_convs[layer_idx](z, target_edge_index)
            
            if self.bn_bool: 
                z = self.up_bns[layer_idx](z)
            
            if n != 1:
                z = self.activation(z)

        # Decoder
        if self.res: z = z + z_res
        z = self.decoder(z)
        return z.unsqueeze(0)