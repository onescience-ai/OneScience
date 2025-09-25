"""Base class for diffusion samplers."""

import abc
from typing import Optional

import xarray

from onescience.flax_models.graphcast import denoisers_base


class Sampler(abc.ABC):
    """A sampling algorithm for a denoising diffusion model.

    This is constructed with a denoising function, and uses it to draw samples.
    """

    _denoiser: denoisers_base.Denoiser

    def __init__(self, denoiser: denoisers_base.Denoiser):
        """Constructs Sampler.

        Args:
          denoiser: A Denoiser which has been trained with an MSE loss to predict
            the noise-free targets.
        """
        self._denoiser = denoiser

    @abc.abstractmethod
    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs
    ) -> xarray.Dataset:
        """Draws a sample using self._denoiser. Contract like Predictor.__call__."""
