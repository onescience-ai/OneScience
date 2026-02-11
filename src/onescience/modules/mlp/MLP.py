import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any, Callable
from collections import OrderedDict
import math

class StandardMLP(nn.Module):
    """
    标准的多层感知机（MLP）实现
    
    特性：
    1. 支持多种激活函数
    2. 支持批量归一化、Dropout、残差连接
    3. 支持自定义权重初始化
    4. 支持多种归一化层
    5. 支持跳过连接（Residual/Skip Connections）
    6. 支持渐进式输出（Progressive Outputs）
    7. 灵活的参数配置
    
    Args:
        input_dim (int): 输入特征维度
        hidden_dims (List[int]): 隐藏层维度列表，例如 [64, 128, 64]
        output_dim (int): 输出特征维度
        activation (str or Callable): 激活函数，支持: 
            'relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', 'selu', 
            'gelu', 'swish', 'mish', 'prelu', 'softplus', 'identity'
        output_activation (str or Callable): 输出层激活函数
        use_bias (bool): 是否使用偏置
        dropout_rate (float): Dropout 概率，0表示不使用
        dropout_position (str): Dropout 位置，'before' 或 'after' 激活函数
        norm_layer (str or None): 归一化层类型，支持: 
            'batch_norm', 'layer_norm', 'instance_norm', 'group_norm', None
        norm_position (str): 归一化位置，'before' 或 'after' 激活函数
        use_skip_connection (bool or str): 是否使用跳过连接，支持: 
            False, 'residual', 'dense', 'highway', 'gated'
        skip_strength (float): 跳过连接强度（0-1之间）
        weight_init (str): 权重初始化方法，支持:
            'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 
            'kaiming_normal', 'orthogonal', 'trunc_normal'
        bias_init (str): 偏置初始化方法，支持: 'zeros', 'ones', 'constant', 'normal'
        last_layer_no_activation (bool): 最后一层是否不使用激活函数
        last_layer_no_norm (bool): 最后一层是否不使用归一化
        progressive_outputs (bool): 是否输出中间层结果
        use_spectral_norm (bool): 是否使用谱归一化
        use_weight_norm (bool): 是否使用权重归一化
        
    Examples:
        >>> # 基础MLP
        >>> mlp = StandardMLP(10, [64, 128, 64], 1, activation='relu')
        >>> 
        >>> # 带有BatchNorm和Dropout的MLP
        >>> mlp = StandardMLP(10, [64, 128, 64], 1, 
        ...                   activation='leaky_relu',
        ...                   dropout_rate=0.2,
        ...                   norm_layer='batch_norm')
        >>> 
        >>> # 带残差连接的MLP
        >>> mlp = StandardMLP(10, [64, 128, 256, 128, 64], 1,
        ...                   activation='gelu',
        ...                   use_skip_connection='residual')
    """
    
    # 激活函数映射
    ACTIVATION_FUNCTIONS = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'swish': lambda: nn.SiLU(),  # Swish = SiLU
        'mish': nn.Mish,
        'prelu': nn.PReLU,
        'softplus': nn.Softplus,
        'identity': nn.Identity,
        'softsign': nn.Softsign,
        'hardtanh': nn.Hardtanh,
        'relu6': nn.ReLU6,
        'celu': nn.CELU,
        'logsigmoid': nn.LogSigmoid,
        'hardshrink': nn.Hardshrink,
        'tanhshrink': nn.Tanhshrink,
        'softshrink': nn.Softshrink,
        'hardsigmoid': nn.Hardsigmoid,
        'hardswish': nn.Hardswish,
    }
    
    # 归一化层映射
    NORM_LAYERS = {
        'batch_norm': nn.BatchNorm1d,
        'layer_norm': nn.LayerNorm,
        'instance_norm': nn.InstanceNorm1d,
        'group_norm': nn.GroupNorm,
        'rms_norm': None,  # 需要自定义实现
    }
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Union[str, Callable] = 'relu',
        output_activation: Union[str, Callable, None] = None,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        dropout_position: str = 'after',  # 'before' or 'after' activation
        norm_layer: Optional[str] = None,
        norm_position: str = 'before',  # 'before' or 'after' activation
        use_skip_connection: Union[bool, str] = False,
        skip_strength: float = 1.0,
        weight_init: str = 'kaiming_uniform',
        bias_init: str = 'zeros',
        last_layer_no_activation: bool = False,
        last_layer_no_norm: bool = False,
        progressive_outputs: bool = False,
        use_spectral_norm: bool = False,
        use_weight_norm: bool = False,
        **kwargs  # 接受额外的参数
    ):
        super().__init__()
        
        # 参数验证
        self._validate_parameters(
            input_dim, hidden_dims, output_dim, dropout_rate, 
            dropout_position, norm_position, skip_strength
        )
        
        # 保存参数
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation if isinstance(activation, str) else 'custom'
        self.output_activation_name = output_activation if isinstance(output_activation, str) else 'custom'
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.dropout_position = dropout_position
        self.norm_layer_type = norm_layer
        self.norm_position = norm_position
        self.use_skip_connection = use_skip_connection
        self.skip_strength = skip_strength
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.last_layer_no_activation = last_layer_no_activation
        self.last_layer_no_norm = last_layer_no_norm
        self.progressive_outputs = progressive_outputs
        self.use_spectral_norm = use_spectral_norm
        self.use_weight_norm = use_weight_norm
        
        # 创建激活函数
        self.activation = self._create_activation(activation)
        self.output_activation = self._create_activation(output_activation) if output_activation else None
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList() if use_skip_connection else None
        
        # 输入层
        in_dim = input_dim
        
        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            linear_layer = nn.Linear(in_dim, hidden_dim, bias=use_bias)
            
            # 应用权重归一化
            if use_weight_norm:
                linear_layer = nn.utils.weight_norm(linear_layer)
            
            # 应用谱归一化
            if use_spectral_norm:
                linear_layer = nn.utils.spectral_norm(linear_layer)
            
            self.layers.append(linear_layer)
            
            # 归一化层
            if norm_layer and (i != len(hidden_dims) - 1 or not last_layer_no_norm):
                norm = self._create_norm_layer(norm_layer, hidden_dim)
                self.norm_layers.append(norm)
            else:
                self.norm_layers.append(nn.Identity())
            
            # Dropout层
            if dropout_rate > 0:
                self.dropout_layers.append(nn.Dropout(dropout_rate))
            else:
                self.dropout_layers.append(nn.Identity())
            
            # 跳过连接层（如果需要）
            if use_skip_connection and in_dim == hidden_dim:
                if use_skip_connection == 'residual':
                    self.skip_connections.append(nn.Identity())
                elif use_skip_connection == 'dense':
                    self.skip_connections.append(nn.Identity())
                elif use_skip_connection == 'highway':
                    gate = nn.Linear(in_dim, hidden_dim)
                    transform = nn.Linear(in_dim, hidden_dim)
                    self.skip_connections.append(nn.ModuleDict({
                        'gate': gate,
                        'transform': transform
                    }))
            
            in_dim = hidden_dim
        
        # 输出层
        output_layer = nn.Linear(in_dim, output_dim, bias=use_bias)
        
        # 输出层的权重归一化
        if use_weight_norm:
            output_layer = nn.utils.weight_norm(output_layer)
        
        # 输出层的谱归一化
        if use_spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
            
        self.layers.append(output_layer)
        
        # 初始化权重
        self._initialize_weights()
        
        # 用于存储中间输出
        self.intermediate_outputs = []
        
        # 额外的配置参数
        self.extra_kwargs = kwargs
    
    def _validate_parameters(self, input_dim, hidden_dims, output_dim, 
                            dropout_rate, dropout_position, 
                            norm_position, skip_strength):
        """验证输入参数"""
        assert input_dim > 0, "输入维度必须大于0"
        assert output_dim > 0, "输出维度必须大于0"
        assert all(dim > 0 for dim in hidden_dims), "所有隐藏层维度必须大于0"
        assert 0 <= dropout_rate <= 1, "Dropout概率必须在0和1之间"
        assert dropout_position in ['before', 'after'], "Dropout位置必须是'before'或'after'"
        assert norm_position in ['before', 'after'], "归一化位置必须是'before'或'after'"
        assert 0 <= skip_strength <= 1, "跳过连接强度必须在0和1之间"
    
    def _create_activation(self, activation: Union[str, Callable, None]) -> nn.Module:
        """创建激活函数"""
        if activation is None:
            return nn.Identity()
        
        if callable(activation):
            # 如果是自定义的激活函数
            try:
                # 尝试实例化（如果是类）
                if isinstance(activation, type):
                    return activation()
                else:
                    # 如果是函数，包装成模块
                    return activation
            except:
                # 如果失败，返回原始的可调用对象
                class CustomActivation(nn.Module):
                    def __init__(self, func):
                        super().__init__()
                        self.func = func
                    
                    def forward(self, x):
                        return self.func(x)
                
                return CustomActivation(activation)
        
        if activation in self.ACTIVATION_FUNCTIONS:
            return self.ACTIVATION_FUNCTIONS[activation]()
        
        # 处理带参数的激活函数
        if activation.startswith('leaky_relu'):
            # 解析negative_slope参数，例如 'leaky_relu_0.2'
            parts = activation.split('_')
            if len(parts) == 3:
                try:
                    negative_slope = float(parts[2])
                    return nn.LeakyReLU(negative_slope=negative_slope)
                except:
                    pass
        
        raise ValueError(f"不支持的激活函数: {activation}")
    
    def _create_norm_layer(self, norm_type: str, dim: int) -> nn.Module:
        """创建归一化层"""
        if norm_type not in self.NORM_LAYERS:
            raise ValueError(f"不支持的归一化层类型: {norm_type}")
        
        norm_class = self.NORM_LAYERS[norm_type]
        
        if norm_class is None and norm_type == 'rms_norm':
            # RMSNorm的自定义实现
            class RMSNorm(nn.Module):
                def __init__(self, dim, eps=1e-8):
                    super().__init__()
                    self.scale = nn.Parameter(torch.ones(dim))
                    self.eps = eps
                
                def forward(self, x):
                    norm = x.norm(2, dim=-1, keepdim=True)
                    return x / (norm + self.eps) * self.scale
            
            return RMSNorm(dim)
        
        # 对于GroupNorm，需要指定组数
        if norm_type == 'group_norm':
            # 默认分组数为32，但如果维度小于32，则使用维度值
            num_groups = min(32, dim)
            if dim % num_groups != 0:
                num_groups = 1  # 如果不能整除，退化为LayerNorm
            return norm_class(num_groups, dim)
        
        # 对于其他归一化层
        return norm_class(dim)
    
    def _initialize_weights(self):
        """初始化权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # 权重初始化
                if self.weight_init == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight)
                elif self.weight_init == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight)
                elif self.weight_init == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif self.weight_init == 'kaiming_normal':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif self.weight_init == 'orthogonal':
                    nn.init.orthogonal_(layer.weight)
                elif self.weight_init == 'trunc_normal':
                    nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
                else:
                    # 默认使用kaiming_uniform
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                
                # 偏置初始化
                if layer.bias is not None:
                    if self.bias_init == 'zeros':
                        nn.init.zeros_(layer.bias)
                    elif self.bias_init == 'ones':
                        nn.init.ones_(layer.bias)
                    elif self.bias_init == 'constant':
                        nn.init.constant_(layer.bias, 0.01)
                    elif self.bias_init == 'normal':
                        nn.init.normal_(layer.bias, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, input_dim] 或 [batch_size, *, input_dim]
            return_intermediate: 是否返回中间层输出
            
        Returns:
            如果 return_intermediate=True: 返回所有层的输出列表
            否则: 返回最后一层的输出
        """
        # 保存原始输入（用于残差连接）
        original_shape = x.shape
        
        # 如果输入是多维的，展平除了最后一个维度之外的所有维度
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        
        # 清空中问输出
        self.intermediate_outputs = []
        
        # 前向传播经过隐藏层
        for i in range(len(self.hidden_dims)):
            # 保存输入（用于残差连接）
            identity = x if (self.use_skip_connection and 
                           x.shape[-1] == self.hidden_dims[i]) else None
            
            # Dropout（如果在激活函数之前）
            if self.dropout_position == 'before' and self.dropout_rate > 0:
                x = self.dropout_layers[i](x)
            
            # 线性变换
            x = self.layers[i](x)
            
            # 归一化（如果在激活函数之前）
            if self.norm_position == 'before':
                x = self.norm_layers[i](x)
            
            # 激活函数
            x = self.activation(x)
            
            # 归一化（如果在激活函数之后）
            if self.norm_position == 'after':
                x = self.norm_layers[i](x)
            
            # Dropout（如果在激活函数之后）
            if self.dropout_position == 'after' and self.dropout_rate > 0:
                x = self.dropout_layers[i](x)
            
            # 应用跳过连接
            if identity is not None:
                if self.use_skip_connection == 'residual':
                    x = identity + self.skip_strength * x
                elif self.use_skip_connection == 'highway' and self.skip_connections:
                    gate = torch.sigmoid(self.skip_connections[i]['gate'](identity))
                    transform = self.skip_connections[i]['transform'](identity)
                    x = gate * x + (1 - gate) * transform
            
            # 保存中间输出
            if return_intermediate or self.progressive_outputs:
                self.intermediate_outputs.append(x.clone())
        
        # 输出层
        x = self.layers[-1](x)
        
        # 输出层激活函数
        if not self.last_layer_no_activation and self.output_activation:
            x = self.output_activation(x)
        
        # 恢复原始形状（如果输入是多维的）
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.output_dim]
            x = x.view(output_shape)
        
        # 保存最终输出
        if return_intermediate or self.progressive_outputs:
            self.intermediate_outputs.append(x.clone())
        
        if return_intermediate:
            return self.intermediate_outputs
        else:
            return x
    
    def get_intermediate_outputs(self) -> List[torch.Tensor]:
        """获取中间层输出"""
        return self.intermediate_outputs
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取所有层的特征图"""
        features = {}
        self.forward(x, return_intermediate=True)
        
        for i, feat in enumerate(self.intermediate_outputs):
            features[f'layer_{i}'] = feat
        
        return features
    
    def get_num_parameters(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """获取层信息"""
        info = []
        
        # 隐藏层信息
        for i, (layer, hidden_dim) in enumerate(zip(self.layers[:-1], self.hidden_dims)):
            layer_info = {
                'type': 'Linear',
                'input_dim': layer.in_features,
                'output_dim': layer.out_features,
                'has_bias': layer.bias is not None,
                'activation': self.activation_name,
                'norm_layer': self.norm_layer_type if i < len(self.norm_layers) else None,
                'dropout': self.dropout_rate if self.dropout_rate > 0 else None
            }
            info.append(layer_info)
        
        # 输出层信息
        output_layer = self.layers[-1]
        output_info = {
            'type': 'Linear',
            'input_dim': output_layer.in_features,
            'output_dim': output_layer.out_features,
            'has_bias': output_layer.bias is not None,
            'activation': self.output_activation_name if not self.last_layer_no_activation else None,
            'norm_layer': None if self.last_layer_no_norm else self.norm_layer_type,
            'dropout': None
        }
        info.append(output_info)
        
        return info
    
    def freeze_layers(self, layer_indices: List[int] = None):
        """冻结指定层"""
        if layer_indices is None:
            # 冻结所有层
            for param in self.parameters():
                param.requires_grad = False
        else:
            # 冻结指定层
            for idx in layer_indices:
                if 0 <= idx < len(self.layers):
                    for param in self.layers[idx].parameters():
                        param.requires_grad = False
    
    def unfreeze_layers(self, layer_indices: List[int] = None):
        """解冻指定层"""
        if layer_indices is None:
            # 解冻所有层
            for param in self.parameters():
                param.requires_grad = True
        else:
            # 解冻指定层
            for idx in layer_indices:
                if 0 <= idx < len(self.layers):
                    for param in self.layers[idx].parameters():
                        param.requires_grad = True


# ==================== 预定义的MLP变体 ====================

class SimpleMLP(StandardMLP):
    """简化的MLP，适用于快速原型开发"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            norm_layer=None,
            use_skip_connection=False
        )


class DeepResMLP(StandardMLP):
    """深度残差MLP，适用于非常深的网络"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout_rate: float = 0.1):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            norm_layer='batch_norm',
            use_skip_connection='residual',
            skip_strength=0.1  # 小权重有助于深度网络训练
        )


class RegularizedMLP(StandardMLP):
    """强正则化的MLP，适用于小数据集"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'elu', dropout_rate: float = 0.3):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            norm_layer='batch_norm',
            use_spectral_norm=True,  # 谱归一化增强稳定性
            weight_init='xavier_normal'
        )


class LightweightMLP(StandardMLP):
    """轻量级MLP，参数少，适合移动设备"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu6'):  # relu6限制激活范围
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            use_bias=False,  # 不使用偏置减少参数
            norm_layer=None,
            dropout_rate=0.0
        )


# ==================== 工厂函数 ====================

def create_mlp(mlp_type: str = 'standard', **kwargs) -> StandardMLP:
    """
    创建MLP的工厂函数
    
    Args:
        mlp_type: MLP类型，支持: 'standard', 'simple', 'residual', 
                 'regularized', 'lightweight', 'deep'
        **kwargs: 传递给MLP构造函数的参数
    
    Returns:
        StandardMLP实例
    """
    mlp_types = {
        'standard': StandardMLP,
        'simple': SimpleMLP,
        'residual': DeepResMLP,
        'regularized': RegularizedMLP,
        'lightweight': LightweightMLP,
        'deep': DeepResMLP,
    }
    
    if mlp_type not in mlp_types:
        raise ValueError(f"未知的MLP类型: {mlp_type}")
    
    return mlp_types[mlp_type](**kwargs)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例1: 基础MLP
    print("示例1: 基础MLP")
    mlp1 = StandardMLP(
        input_dim=784,
        hidden_dims=[256, 128, 64],
        output_dim=10,
        activation='relu',
        dropout_rate=0.2
    )
    print(f"参数数量: {mlp1.get_num_parameters():,}")
    print(mlp1)
    # 示例2: 带残差连接的深度MLP
    print("\n示例2: 带残差连接的深度MLP")
    mlp2 = DeepResMLP(
        input_dim=512,
        hidden_dims=[512, 512, 512, 512, 512],  # 非常深
        output_dim=100,
        activation='gelu'
    )
    
    # 示例3: 使用工厂函数创建
    print("\n示例3: 使用工厂函数创建")
    mlp3 = create_mlp(
        mlp_type='regularized',
        input_dim=100,
        hidden_dims=[200, 100, 50],
        output_dim=1,
        activation='leaky_relu'
    )
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 784)
    
    # 获取中间输出
    outputs = mlp1(x, return_intermediate=True)
    print(f"\n输入形状: {x.shape}")
    print(f"输出层数: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  第{i}层输出形状: {out.shape}")
    
    # 获取层信息
    print("\nMLP层信息:")
    for i, layer_info in enumerate(mlp1.get_layer_info()):
        print(f"  层{i}: {layer_info}")
