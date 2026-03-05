import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp

class XiheMlp(nn.Module):
    """
    全局 SIE 的第二步：组间传播模块，在 group 空间内进行信息交互与融合。

    Args:
        dim (int): 输入通道数 C。
        num_groups (int): group vectors 数量 G，默认为 32。
        mlp_ratio (float): MLP 隐层扩展比例，默认为 4.0。
        drop (float): MLP 的 dropout 比例，默认为 0.0。
        act_layer (nn.Module): 激活函数层类型，默认为 nn.GELU。
        LN (nn.Module): 归一化层类型，默认为 nn.LayerNorm。

    形状:
        输入 x (torch.Tensor): 输入 group vectors，形状为 (B, G, C)
        输出 y (torch.Tensor): 传播融合后的 group vectors，形状为 (B, G, C)

    Example:        >>> group_prop = XiheMlp(
        ...     dim=192,
        ...     num_groups=32,
        ...     mlp_ratio=4.0
        ... )
        >>> B, G, C = 2, 32, 192
        >>> x = torch.randn(B, G, C)
        >>> out = group_prop(x)
        >>> out.shape
        torch.Size([2, 32, 192])
    """

    def __init__(
        self,
        dim, 
        num_groups=32,
        mlp_ratio=4.0,
        drop=0.0, 
        act_layer=nn.GELU,
        LN=nn.LayerNorm
        ):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups

        # LayerNorm
        self.norm1 = LN(dim)
        self.norm2 = LN(dim)
        
        # Token-mixing MLP (在 group 维度上传播信息)
        mlp_token_dim = int(num_groups * mlp_ratio)
        self.mlp_token = Mlp(
            in_features=num_groups,
            hidden_features=mlp_token_dim,
            act_layer=act_layer,
            drop=drop,
        )        
    
       # Channel-mixing MLP (在 embedding 维度融合特征)
        mlp_channel_dim = int(dim * mlp_ratio)
        self.mlp_channel =Mlp(
            in_features=dim,
            hidden_features=mlp_channel_dim,
            act_layer=act_layer,
            drop=drop,
        )
        
    def forward(self, x):
        """
        x: (B, G, C) -> 输入 group vectors
        """
        B, G, C = x.shape
        shortcut=x
        

        # Step 1: Token mixing (group 维度传播信息)
        x = self.norm1(x)          # (B, G, C)
        x = x.transpose(1, 2)            # (B, C, G)
        x = self.mlp_token(x)            # (B, C, G) 先对group进行mlp
        x = x.transpose(1, 2)            # (B, G, C)
        x = shortcut + x              # 残差连接
        # Step 2: Channel mixing (embedding 维度融合)  
        y = self.norm2(x)            # (B, G, C)
        y = self.mlp_channel(y)          # (B, G, C) 在对channel进行mlp
        y = x + y              # 残差连接
        return y