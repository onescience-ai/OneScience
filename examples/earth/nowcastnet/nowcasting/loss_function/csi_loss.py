import torch
import torch.nn as nn


class CSILoss(nn.Module):
    def __init__(self, configs):
        super(CSILoss, self).__init__()
        self.configs = configs

    def prep_clf(self, obs, pre, threshold=0.1):
        obs = torch.where(obs >= threshold, 1, 0)
        pre = torch.where(pre >= threshold, 1, 0)
        # True positive (TP)
        hits = torch.sum((obs == 1) & (pre == 1))
        # False negative (FN)
        misses = torch.sum((obs == 1) & (pre == 0))
        # False positive (FP)
        falsealarms = torch.sum((obs == 0) & (pre == 1))
        # True negative (TN)
        correctnegatives = torch.sum(
            (obs == 0) & (pre == 0))
        return hits, misses, falsealarms, correctnegatives

    def forward(self, obs, pre, threshold=0.1):
        hits, misses, falsealarms, correctnegatives = self.prep_clf(
            obs=obs, pre=pre, threshold=threshold
        )
        return 1 - hits / (hits + falsealarms + misses)
