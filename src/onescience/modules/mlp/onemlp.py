import torch

from torch import nn
from .meshgraphmlp import MeshGraphMLP

class OneMlp(MeshGraphMLP):
    def __init__(self, style="MeshGraphMLP"):
        if style == "MeshGraphMLP":
            self.MeshGraphMLP = MeshGraphMLP(input_dim=1, output_dim=1, hidden_dim=1, hidden_layers=None, activation_fn=nn.ReLU())
        else:
            raise NotImplementedError