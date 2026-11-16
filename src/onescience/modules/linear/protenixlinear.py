"""
Protenix Linear Modules
Implements linear layers with customized initialization for Protenix (AlphaFold3)
"""
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.models.openfold.primitives import ProtenixLayerNorm, trunc_normal_init_


class ProtenixLinear(nn.Linear):
    """
    Linear module with customized initialization for Protenix.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: Whether to add bias. Defaults to True.
        precision: Optional precision for computation. Defaults to None.
        initializer: Initialization method. Choose from ['default', 'relu', 'zeros'].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        precision: torch.dtype = None,
        initializer: str = "default",
    ):
        self.use_bias = bias
        self.precision = precision
        self.initializer = initializer
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._init_params()

    @torch.no_grad()
    def _init_params(self):
        if self.use_bias:
            nn.init.zeros_(self.bias)

        if self.initializer == "default":
            trunc_normal_init_(self.weight, scale=1.0)
        elif self.initializer == "relu":
            trunc_normal_init_(self.weight, scale=2.0)
        elif self.initializer == "zeros":
            nn.init.zeros_(self.weight)
        else:
            raise ValueError(f"Invalid initializer: {self.initializer}.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.precision is not None:
            input_dtype = input.dtype
            with torch.cuda.amp.autocast(enabled=False):
                bias = (
                    self.bias.to(dtype=self.precision)
                    if self.bias is not None
                    else None
                )
                return F.linear(
                    input.to(dtype=self.precision),
                    self.weight.to(dtype=self.precision),
                    bias,
                ).to(dtype=input_dtype)
        else:
            return F.linear(input, self.weight, self.bias)


class ProtenixLinearNoBias(ProtenixLinear):
    """Linear layer without bias for Protenix."""
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, bias=False, **kwargs)


class ProtenixBiasInitLinear(ProtenixLinear):
    """
    Linear layer with bias initialization support.
    Called just like torch.nn.Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        biasinit: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )
        nn.init.zeros_(tensor=self.weight)
        if bias:
            nn.init.constant_(tensor=self.bias, val=biasinit)
