import torch
import torch.nn as nn
import numpy as np
from timm.layers import trunc_normal_
from einops import rearrange, repeat
# 导入模块化组件
from onescience.modules.block.Transolver_block import Transolver_block
from onescience.modules import OneMlp

class Transolver3D_plus(nn.Module):
    """
    Transolver3D_plus 模型。

    这是 Transolver3D 的增强版本。
    它使用了 'unstructured_plus' 类型的注意力机制，
    这种机制引入了更复杂的特征交互（如 Gumbel Softmax 加权切片），旨在提升对三维不规则网格物理场的建模能力。

    Args:
        space_dim (int): 空间维度。默认值: 1。
        n_layers (int): Block 层数。默认值: 5。
        n_hidden (int): 隐藏层维度。默认值: 256。
        dropout (float): Dropout 概率。默认值: 0。
        n_head (int): 注意力头数。默认值: 8。
        act (str): 激活函数。默认值: 'gelu'。
        mlp_ratio (float): MLP 膨胀比率。默认值: 1。
        fun_dim (int): 输入物理场特征维度。默认值: 1。
        out_dim (int): 输出特征维度。默认值: 1。
        slice_num (int): 切片数量。默认值: 32。
        ref (int): 统一位置编码参考分辨率。默认值: 8。
        unified_pos (bool): 是否使用统一位置编码。默认值: False。
    """
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super(Transolver3D_plus, self).__init__()
        self.__name__ = 'Transolver3D'
        self.ref = ref
        self.unified_pos = unified_pos
        
        if self.unified_pos:
            input_dim = fun_dim + self.ref * self.ref * self.ref
        else:
            input_dim = fun_dim + space_dim

        self.preprocess = OneMlp(
            style="StandardMLP",
            input_dim=input_dim,
            hidden_dims=[n_hidden * 2], 
            output_dim=n_hidden,
            activation=act,
            use_bias=True,
            use_skip_connection=False
        )
        
        self.n_hidden = n_hidden
        self.space_dim = space_dim

        # === Transolver Blocks ===
        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=(_ == n_layers - 1),
                geotype='unstructured_plus' 
            )
            for _ in range(n_layers)
        ])
        
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

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

    def get_grid(self, my_pos):
        batchsize = my_pos.shape[0]

        # 3D Grid Generation Logic
        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(my_pos.device).reshape(batchsize, self.ref ** 3, 3)

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):
        cfd_data = data
        x, fx, T = cfd_data.x, None, None
        x = x[None, :, :]
        
        if self.unified_pos:
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx[0]