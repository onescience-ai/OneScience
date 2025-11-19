import torch

def L1_loss(x, y):
    return torch.nn.functional.l1_loss(x, y)