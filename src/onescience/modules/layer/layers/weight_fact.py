import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def weight_fact(w, mean=1.0, stddev=0.1):
    """
    Randomly factorize the weight matrix into a product of vectors and a matrix

    Parameters
    ----------
    w : torch.Tensor
    mean : float, optional, default=1.0, mean of the normal distribution to sample the random scale factor
    stddev: float, optional, default=0.1, standard deviation of the normal distribution to sample the random scale factor
    """

    g = torch.normal(mean, stddev, size=(w.shape[0], 1))
    g = torch.exp(g)
    v = w / g
    return g, v


class WeightFactLinear(nn.Module):
    """Weight Factorization Layer for 2D Tensors, more details in https://arxiv.org/abs/2210.01274

    Parameters
    ----------
    in_features : int
        Size of the input features
    out_features : int
        Size of the output features
    bias : bool, optional
        Apply the bias to the output of linear layer, by default True
    reparam : dict, optional
        Dictionary with the mean and standard deviation to reparametrize the weight matrix,
        by default {'mean': 1.0, 'stddev': 0.1}

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mean: float = 1.0,
        stddev=0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean = mean
        self.stddev = stddev

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Factorize weights and reset bias"""
        nn.init.xavier_uniform_(self.weight)
        g, v = weight_fact(self.weight.detach(), mean=self.mean, stddev=self.stddev)
        self.g = nn.Parameter(g)
        self.v = nn.Parameter(v)
        self.weight = None  # remove the weight parameter

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.g * self.v
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        """Print information about weight factorization"""
        return (
            "in_features={}, out_features={}, bias={}, mean = {}, stddev = {}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.mean,
                self.stddev,
            )
        )
