from typing import List, Dict, Optional

import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import AutoCfdModel

from onescience.modules import OneFourier
torch.manual_seed(0)
np.random.seed(0)

class FnoBlock(nn.Module):
    """
    FNO 块 (Fourier Neural Operator Block)。
    
    包含一个 2D 频域卷积路径 (通过 OneFourier 调用) 和一个 1x1 的空间卷积残差路径，
    最后加上可选的激活函数。
    """
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        modes1: int,
        modes2: int,
        act_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.modes1 = modes1
        self.modes2 = modes2
        self.act_fn = act_fn


        self.conv0 = OneFourier(
            style="FNOSpectralConv2d",
            in_channels=self.in_chan,
            out_channels=self.out_chan,
            modes1=self.modes1,
            modes2=self.modes2
        )
        
        self.w0 = nn.Conv2d(self.in_chan, self.out_chan, 1)

    def forward(self, x: Tensor) -> Tensor:

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        
        if self.act_fn is not None:
            x = self.act_fn(x)
            
        return x


class Fno2d(AutoCfdModel):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        n_case_params: int,
        loss_fn: nn.Module,
        num_layers: int,
        modes1: int = 12,
        modes2: int = 12,
        hidden_dim: int = 20,
        padding: Optional[int] = None,
    ):
        super().__init__(loss_fn)

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1
            and self.fc2 .
        """
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.num_layers = num_layers
        self.modes1 = modes1
        self.modes2 = modes2
        self.hidden_dim = hidden_dim
        self.padding = padding  # pad the domain if input is non-periodic

        self.act_fn = nn.GELU()
        # Channel projection into `hidden_dim` channels
        # +1 for mask, +2 for coordinates
        self.fc0 = nn.Conv2d(
            in_chan + 1 + 2 + n_case_params,
            self.hidden_dim,
            1,
            1,
            0,
        )
        # input channel is 12: the solution of the previous 10
        # timesteps + 2 locations
        # (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        # FNO blocks
        blocks = []
        for _ in range(self.num_layers):
            blocks.append(
                FnoBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.modes1,
                    self.modes2,
                    self.act_fn,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.fc1 = nn.Conv2d(self.hidden_dim, 128, 1, 1, 0)
        self.fc2 = nn.Conv2d(128, self.out_chan, 1, 1, 0)

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
    ) -> Dict:
        """
        Args:
        - input: (b, c, h, w)
        - labels: (b, c, h, w)
        - mask: (b, h, w), a binary mask for indicating the geometry
                1 for interior, 0 for obstacles.

        Returns: a tensor of shape (b, c, h, w), the solution at
            the next timestep
        """
        batch_size, n_chan, height, width = inputs.shape

        if mask is None:
            # When there is no mask, we assume that there is no obstacles.
            mask = torch.ones((batch_size, 1, height, width)).to(inputs.device)
        else:
            if mask.dim() == 3:  # (B, h, w)
                mask = mask.unsqueeze(1)  # (B, 1, h, w)
        inputs = torch.cat([inputs, mask], dim=1)  # (B, c + 1, h, w)

        # Physical properties
        props = case_params  # (B, p)
        props = props.unsqueeze(-1).unsqueeze(-1)  # (B, p, 1, 1)
        props = props.repeat(1, 1, height, width)  # (B, p, H, W)

        # Append (x, y) coordinates to every location
        grid = self.get_coords(inputs.shape, inputs.device)  # (b, 2, h, w)
        inputs = torch.cat(
            (inputs, grid, props), dim=1
        )  # (b, c + 2 + 2, h, w)

        # Project channels
        inputs = self.fc0(inputs)  # (b, hidden_dim, h, w)
        # x = x.permute(0, 3, 1, 2)  # (b, c, h, w)?
        if self.padding is not None:
            # pad the domain if input is non-periodic
            inputs = F.pad(inputs, [0, self.padding, 0, self.padding])

        inputs = self.blocks(inputs)  # (b, hidden_dim, h, w)
        if self.padding is not None:
            # pad the domain if inputis non-periodic
            inputs = inputs[..., : -self.padding, : -self.padding]

        inputs = self.fc1(inputs)  # (b, 128, h, w)
        inputs = self.act_fn(inputs)
        preds = self.fc2(inputs)  # (b, c_out, h, w)

        # Masked locations are not prediction
        preds = preds * mask

        if label is not None:
            label = label * mask
            loss = self.loss_fn(preds=preds, labels=label)
            return dict(
                preds=preds,
                loss=loss,
            )
        return dict(preds=preds)

    def get_coords(self, shape, device):
        """
        Return a tensor of shape (b, 2, h, w) such that the element at
        [:, :, i, j] is the (x, y) coordinates at the grid location (i, j).
        """
        bsz, c, size_x, size_y = shape
        grid_x = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid_x = grid_x.reshape(1, 1, size_x, 1).repeat([bsz, 1, 1, size_y])
        grid_y = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        grid_y = grid_y.reshape(1, 1, 1, size_y).repeat([bsz, 1, size_x, 1])
        coords = torch.cat([grid_x, grid_y], dim=1).to(device)  # (b, 2, h, w)
        return coords

    def generate(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        outputs = self.forward(
            inputs=inputs, case_params=case_params, mask=mask
        )  # (b, c, h, w)
        preds = outputs["preds"]
        return preds

    def generate_many(
        self, inputs: Tensor, case_params: Tensor, mask: Tensor, steps: int
    ) -> List[Tensor]:
        """
        Args:
            x (Tensor): (c, h, w)
            case_params (Tensor): (p)
            mask (Tensor): (h, w), 1 for interior, 0 for obstacles.
        Returns:
            output: (steps, c, h, w)
        """
        assert len(inputs.shape) == len(case_params.shape) + 2
        if inputs.dim() == 3:
            # Add a dimension for batch size of 1
            inputs = inputs.unsqueeze(0)
            case_params = case_params.unsqueeze(0)
            mask = mask.unsqueeze(0)
        assert inputs.shape[0] == case_params.shape[0] == mask.shape[0]

        cur_frame = inputs  # (b, c, h, w)
        preds = []
        for _ in range(steps):
            cur_frame = self.generate(
                inputs=cur_frame, case_params=case_params, mask=mask
            )
            preds.append(cur_frame)
        return preds
