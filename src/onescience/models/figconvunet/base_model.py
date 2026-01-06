from typing import Any, Dict, Tuple

import torch

from onescience.models.module import Module


class BaseModel(Module):
    """Base model class."""

    def data_dict_to_input(self, data_dict, **kwargs) -> Any:
        """Convert data dictionary to appropriate input for the model."""
        raise NotImplementedError

    def loss_dict(self, data_dict, **kwargs) -> Dict:
        """Compute the loss dictionary for the model."""
        raise NotImplementedError

    @torch.no_grad()
    def eval_dict(self, data_dict, **kwargs) -> Dict:
        """Compute the evaluation dictionary for the model."""
        raise NotImplementedError

    def image_pointcloud_dict(self, data_dict, datamodule) -> Tuple[Dict, Dict]:
        """Compute the image dict and pointcloud dict for the model."""
        raise NotImplementedError
