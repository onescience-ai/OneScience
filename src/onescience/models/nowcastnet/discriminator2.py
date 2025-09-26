"""
discrimination.py:
This is a completely original code, reproducing the discriminator part of the model.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.utils.layers.discrimination.DBlock import LastConv, ProjBlock


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.BlockDown_2D = nn.Conv2d(
            29, 64, kernel_size=(9, 9), stride=(2, 2), padding=4
        )

        self.BlockDown_3D_1 = nn.Conv3d(
            1, 4, kernel_size=(4, 9, 9), stride=(1, 2, 2), padding=(0, 4, 4)
        )
        self.BlockDown_3D_2 = nn.Conv3d(
            4, 8, kernel_size=(4, 9, 9), stride=(1, 2, 2), padding=(0, 4, 4)
        )

        self.LBlockDown_1 = ProjBlock(208, 128)
        self.LBlockDown_2 = ProjBlock(128, 256)
        self.LBlockDown_3 = ProjBlock(256, 512)
        self.LBlockDown_4 = ProjBlock(512, 512)

        self.lastconv = LastConv(512, 1)

    def forward(self, x):
        x1 = self.BlockDown_2D(x)

        x_c = x.unsqueeze(axis=1)
        x2 = self.BlockDown_3D_1(x_c)

        x_d = F.pad(x, (0, 0, 0, 0, 1, 2),
                    mode="constant", value=0)
        x_d = x_d.view(
            x_d.shape[0], 4, 8, x_d.shape[2], x_d.shape[3])
        x3 = self.BlockDown_3D_2(x_d)

        x2 = x2.view(x2.shape[0], -1,
                     x2.shape[3], x2.shape[4])
        x3 = x3.view(x3.shape[0], -1,
                     x3.shape[3], x3.shape[4])
        x_temp = torch.cat((x1, x2, x3), dim=1)

        x_temp = self.LBlockDown_1(x_temp)
        x_temp = self.LBlockDown_2(x_temp)
        x_temp = self.LBlockDown_3(x_temp)
        x_temp = self.LBlockDown_4(x_temp)

        out = self.lastconv(x_temp)

        return out
