"""Mamba RNN 模块 - 基于状态空间模型的高效序列建模"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class S6(nn.Module):
    """选择性状态空间模型 (Selective SSM) - Mamba 核心组件"""
    def __init__(self, d_model, d_state=16, dt_rank='auto', expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank != 'auto' else d_model // 16
        self.d_inner = d_model * expand

        # 投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)

        # SSM 参数
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 状态空间参数
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape

        # 输入投影和门控
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # 1D 卷积
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)

        # SSM 参数计算
        x_dbl = self.x_proj(x)
        dt, B_ssm, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        # 选择性扫描
        A = -torch.exp(self.A_log.float())
        y = self.selective_scan(x, dt, A, B_ssm, C, self.D)

        # 门控和输出投影
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def selective_scan(self, u, delta, A, B, C, D):
        """选择性扫描算法"""
        B_batch, L, D_inner = u.shape

        # 离散化
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        # 扫描
        x = torch.zeros(B_batch, D_inner, self.d_state, device=u.device)
        ys = []

        for i in range(L):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum('bdn,bn->bd', x, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y


class MambaBlock(nn.Module):
    """Mamba 块 - 包含 SSM 和残差连接"""
    def __init__(self, d_model, d_state=16, expand=2, dt_rank='auto'):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = S6(d_model, d_state, dt_rank, expand)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class Mamba(nn.Module):
    """完整的 Mamba 模型"""
    def __init__(self, d_model, n_layers, d_state=16, expand=2, vocab_size=50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, expand) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits
