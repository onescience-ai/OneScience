from typing import Optional, Protocol
import xarray


class Denoiser(Protocol):
  """A denoising model that conditions on inputs as well as noise level."""

  def __call__(
      self,
      inputs: xarray.Dataset,
      noisy_targets: xarray.Dataset,
      noise_levels: xarray.DataArray,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> xarray.Dataset:
    """Computes denoised targets from noisy targets.

    Args:
      inputs: Inputs to condition on, as for Predictor.__call__.
      noisy_targets: Targets which have had i.i.d. zero-mean Gaussian noise
        added to them (where the noise level used may vary along the 'batch'
        dimension).
      noise_levels: A DataArray with dimensions ('batch',) specifying the noise
        levels that were used for each example in the batch.
      forcings: Optional additional per-target-timestep forcings to condition
        on, as for Predictor.__call__.
      **kwargs: Any additional custom kwargs.

    Returns:
      Denoised predictions with the same shape as noisy_targets.
    """
