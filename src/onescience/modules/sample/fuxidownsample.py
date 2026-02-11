import torch
from torch import nn



class FuXiDownSample(nn.Module):
    def __init__(self, 
                 in_chans=1536, 
                 out_chans=1536, 
                 num_groups=32, 
                 num_residuals= 2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        res = x + shortcut
        if h % 2 != 0:
            res = res[:, :, :-1, :]
        if w % 2 != 0:
            res = res[:, :, :, :-1]
        return res

