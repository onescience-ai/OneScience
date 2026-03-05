
import torch
import torch.nn as nn
import torch_geometric.nn as nng
import random
from onescience.modules import OneMlp, OneSample

class Model(nn.Module):
    """
    Graph U-Net 模型。
    
    基于图神经网络的 U-Net 结构，包含 Encoder-Decoder 和 Skip Connections。
    使用 SpatialGraphDownsample 进行图池化，使用 SpatialGraphUpsample 进行反池化。
    """
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
        self.__name__ = "Graph_UNet"
        
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
        
        self.encoder = OneMlp(
            style="StandardMLP",
            input_dim=args.fun_dim,
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
            self.down_samples.append(
                OneSample(
                    style="SpatialGraphDownsample",
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
        
        # --- 3. Upsample Module (使用 OneSample) ---
        self.up_sampler = OneSample(style="SpatialGraphUpsample")
        
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

    def forward(self, x, fx, T=None, geo=None):
        if geo is None: raise ValueError("Edge index required")
        if fx.dim() == 3: fx = fx.squeeze(0)
        if geo.dim() == 3: edge_index = geo.squeeze(0)
        else: edge_index = geo
        
        # Encoder
        z = self.encoder(fx)
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
        
        # Assuming x contains coords in first 2 columns as per original code logic
        current_pos = x[:, :2] 
        pos_history.append(current_pos.clone())

        # Levels 1 to L-1
        for n in range(self.L - 1):
            z, current_pos, edge_index, _ = self.down_samples[n](z, current_pos, edge_index)
            
            pos_history.append(current_pos.clone())
            edge_index_history.append(edge_index.clone())

            z = self.down_convs[n+1](z, edge_index)
            if self.bn_bool: z = self.down_bns[n+1](z)
            z = self.activation(z)
            
            skip_connections.append(z.clone())
        
        # Up Path
        for n in range(self.L - 1, 0, -1):
            layer_idx = n - 1
            
            pos_low = pos_history[n]
            pos_high = pos_history[n-1]
            z_skip = skip_connections[n-1]
            
            target_edge_index = edge_index_history[n-1]

            z = self.up_sampler(z, pos_low, pos_high)
            
            z = torch.cat([z, z_skip], dim=1)
            
            z = self.up_convs[layer_idx](z, target_edge_index)
            
            if self.bn_bool: 
                z = self.up_bns[layer_idx](z)
            
            if n != 1:
                z = self.activation(z)

        # Decoder
        if self.res: z = z + z_res
        z = self.decoder(z)
        return z.unsqueeze(0)