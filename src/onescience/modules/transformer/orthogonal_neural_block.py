import torch
import torch.nn as nn
import warnings
from onescience.modules.mlp.onemlp import OneMlp
from onescience.modules.attention.oneattention import OneAttention

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """
    计算矩阵 A 的 Cholesky 分解。如果 A 只是半正定 (p.s.d)，则在对角线上添加微小的抖动 (jitter) 以保证数值稳定性。
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        if torch.isnan(L).any():
            raise RuntimeError
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise ValueError(f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN.")
        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10**i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(f"A not p.d., added jitter of {jitter_new} to the diagonal", RuntimeWarning)
                return L
            except RuntimeError:
                continue
        raise e

class OrthogonalNeuralBlock(nn.Module):
    """
    正交神经 Transformer 块 (Orthogonal Neural Block)。

    该模块是 Orthogonal Neural Operator (ONO) 的核心组件。它通过双分支架构进行更新：
    1. **特征分支 (x)**: 使用注意力机制（支持 Nystrom 等线性近似）和 MLP 进行非线性特征提取。
    2. **物理分支 (fx)**: 在特征分支的引导下，通过维护协方差矩阵 (Covariance Matrix) 
       并利用 Cholesky 分解进行正交化投影，从而在物理信号的传播过程中保持特征的独立性和正交性，有效缓解了深层网络的过度平滑问题。

    Args:
        num_heads (int): 注意力头的数量。
        hidden_dim (int): 隐藏层特征维度。
        dropout (float): Dropout 概率。
        act (str, optional): 激活函数类型。默认值: "gelu"。
        attn_type (str, optional): 注意力类型，可选 "nystrom", "linear", "selfAttention"。默认值: "nystrom"。
        mlp_ratio (int, optional): MLP 隐藏层扩展倍数。默认值: 4。
        last_layer (bool, optional): 是否为最后一层。如果是，输出前馈网络将替换为简单线性投影。默认值: False。
        momentum (float, optional): 协方差矩阵的动量更新率。默认值: 0.9。
        psi_dim (int, optional): 投影到正交空间的维度。默认值: 8。
        out_dim (int, optional): 最后一层的输出维度。默认值: 1。

    形状:
        输入 x: (B, N, C) - 用于更新注意力和生成投影矩阵的辅助特征。
        输入 fx: (B, N, C) - 核心物理场特征。
        输出: Tuple[Tensor, Tensor]，返回更新后的 (x, fx)，形状与输入一致 (除非 last_layer=True 时 fx 的通道数改变)。

    Example:
        >>> block = OrthogonalNeuralBlock(num_heads=4, hidden_dim=64, dropout=0.1)
        >>> x = torch.randn(2, 1024, 64)
        >>> fx = torch.randn(2, 1024, 64)
        >>> out_x, out_fx = block(x, fx)
        >>> print(out_fx.shape)
        torch.Size([2, 1024, 64])
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        attn_type="nystrom",
        mlp_ratio=4,
        last_layer=False,
        momentum=0.9,
        psi_dim=8,
        out_dim=1,
    ):
        super().__init__()
        self.momentum = momentum
        self.psi_dim = psi_dim

        self.register_buffer("feature_cov", torch.zeros(psi_dim, psi_dim))
        self.register_parameter("mu", nn.Parameter(torch.zeros(psi_dim)))
        self.ln_1 = nn.LayerNorm(hidden_dim)
        
        if attn_type == "nystrom":
            attn_style = "NystromAttention"
        elif attn_type == "linear":
            attn_style = "LinearAttention"
        elif attn_type == "selfAttention":
            attn_style = "SelfAttention"
        else:
            raise ValueError("Attn type only supports nystrom, linear, or selfAttention")
            
        attn_kwargs = {
            "dim": hidden_dim,
            "heads": num_heads,
            "dim_head": hidden_dim // num_heads,
            "dropout": dropout,
        }
        if attn_type == "linear":
            attn_kwargs["attn_type"] = "galerkin"
            
        self.Attn = OneAttention(style=attn_style, **attn_kwargs)
            
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        self.mlp = OneMlp(
            style="StandardMLP",
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim * mlp_ratio],
            activation=act,
            use_bias=True
        )
        
        self.proj = nn.Linear(hidden_dim, psi_dim)
        self.ln_3 = nn.LayerNorm(hidden_dim)
        
        if last_layer:
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.mlp2 = OneMlp(
                style="StandardMLP",
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim * mlp_ratio],
                activation=act,
                use_bias=True
            )

    def forward(self, x, fx):
        x = self.Attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        x_ = self.proj(x)
        
        if self.training:
            batch_cov = torch.einsum("blc, bld->cd", x_, x_) / x_.shape[0] / x_.shape[1]
            with torch.no_grad():
                self.feature_cov.mul_(self.momentum).add_(
                    batch_cov, alpha=1 - self.momentum
                )
        else:
            batch_cov = self.feature_cov
            
        L = psd_safe_cholesky(batch_cov)
        L_inv_T = L.inverse().transpose(-2, -1)
        x_ = x_ @ L_inv_T

        fx = (x_ * torch.nn.functional.softplus(self.mu)) @ (
            x_.transpose(-2, -1) @ fx
        ) + fx
        fx = self.mlp2(self.ln_3(fx))
        return x, fx