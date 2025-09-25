from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class ChannelMetadata:
    """Metadata describing a data channel."""

    name: str
    level: str = ""
    auxiliary: bool = False


class DownscalingDataset(torch.utils.data.Dataset, ABC):
    """An abstract class that defines the interface for downscaling datasets."""

    @abstractmethod
    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""

    @abstractmethod
    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""

    @abstractmethod
    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""

    @abstractmethod
    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""

    @abstractmethod
    def time(self) -> List:
        """Get time values from the dataset."""

    @abstractmethod
    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return x

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return x

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x

    def info(self) -> dict:
        """Get information about the dataset."""
        return {}
