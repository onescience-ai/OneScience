import torch
from torch import Tensor

from onescience.utils.graphcast.graph_utils import deg2rad


def normalized_grid_cell_area(lat: Tensor, unit="deg") -> Tensor:
    """Normalized area of the latitude-longitude grid cell"""
    if unit == "deg":
        lat = deg2rad(lat)
    area = torch.abs(torch.cos(lat))
    return area / torch.mean(area)
