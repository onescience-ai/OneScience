import torch
import torch.nn as nn
from .spectralNormalization import  SpectralNorm
from torch.nn.utils import spectral_norm


class DBlockDown(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(DBlockDown, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 =SpectralNorm( nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride=1,padding=1))
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)) 
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out


class DBlockDownFirst(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(DBlockDownFirst, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride=1,padding=1))
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1) ) 
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)

        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out



class DBlock(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3,stride=1,padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x)
        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        out = x1 + x2
        return out


class DBlock3D_1(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)

        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2

        return out

class DBlock3D_2(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_2, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1,stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2

        return out


class LBlockDown(nn.Module):
    def   __init__(self, in_channels, out_channels):
        super(LBlockDown, self).__init__()
        self.up_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        )
        self.down_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)),
        )

    def forward(self, x):
        x1 = self.up_conv(x)
        x2 = self.down_conv(x)
        out = x1 + x2
        return out

class ProjBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ProjBlock, self).__init__()
        self.one_conv = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))
        self.double_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        )
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.one_conv(x)
        x1 = self.maxpool(x1)

        x2 = self.double_conv(x)
        x2 = self.maxpool(x2)

        output = x1 + x2
        return output


class LastConv(nn.Module):
    def   __init__(self, in_channels, out_channels):
        super(LastConv, self).__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(inplace=False),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3)),
        )

    def forward(self, x):
        out = self.conv(x)
        return out