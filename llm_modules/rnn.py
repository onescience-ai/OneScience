"""经典 RNN 模块"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM 层"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, hidden=None):
        return self.lstm(x, hidden)


class GRU(nn.Module):
    """GRU 层"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, hidden=None):
        return self.gru(x, hidden)


class BiLSTM(nn.Module):
    """双向 LSTM"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           dropout=dropout, bidirectional=True, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output


class RWKV(nn.Module):
    """RWKV 时间混合层"""
    def __init__(self, d_model):
        super().__init__()
        self.time_decay = nn.Parameter(torch.ones(d_model))
        self.time_first = nn.Parameter(torch.ones(d_model))

        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        xx = torch.cat([torch.zeros(B, 1, C, device=x.device), x[:, :-1]], dim=1)

        k = self.key(x * self.time_mix_k + xx * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + xx * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + xx * (1 - self.time_mix_r))

        wkv = self._wkv(k, v)
        rwkv = torch.sigmoid(r) * wkv

        return self.output(rwkv)

    def _wkv(self, k, v):
        B, T, C = k.shape
        w = -torch.exp(self.time_decay)
        u = self.time_first

        out = torch.zeros_like(k)
        state = torch.zeros(B, C, device=k.device)

        for t in range(T):
            kt = k[:, t]
            vt = v[:, t]

            wkv = (state + u * kt) / (state.abs().sum(dim=-1, keepdim=True) + torch.exp(u + kt).sum(dim=-1, keepdim=True))
            out[:, t] = wkv * vt

            state = state * torch.exp(w) + torch.exp(kt) * vt

        return out
