import numpy as np
import torch


def set_seed(seed: int = 0):
    """Random seed initialization."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
