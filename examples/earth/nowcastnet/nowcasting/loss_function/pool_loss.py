import torch
import torch.nn as nn

class MaxPoolLoss(nn.Module):
    def __init__(self, configs):
        super(MaxPoolLoss, self).__init__()
        self.configs = configs
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=1)

    def forward(self, true_image, fake_image):

        output1 = self.pool1(true_image)
        output2 = self.pool2(fake_image)

        weight = torch.minimum(torch.full(output1.shape, 24).to(self.configs.device), 1 + output1)
        loss = torch.mean(torch.abs(output1 - output2) * weight)

        return loss