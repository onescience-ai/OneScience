"""Wrappers for Predictors which allow them to work with data cleaned of NaNs.

The Predictor which is wrapped sees inputs and targets without NaNs, and makes
NaNless predictions.
"""

from typing import Optional, Tuple

import numpy as np
import xarray

from onescience.flax_models.graphcast import predictor_base as base


class NaNCleaner(base.Predictor):
    """A predictor wrapper than removes NaNs from ingested data.

    The Predictor which is wrapped sees inputs and targets without NaNs.
    """

    def __init__(
        self,
        predictor: base.Predictor,
        var_to_clean: str,
        fill_value: xarray.Dataset,
        reintroduce_nans: bool = False,
    ):
        """Initializes the NaNCleaner."""
        self._predictor = predictor
        self._fill_value = fill_value[var_to_clean]
        self._var_to_clean = var_to_clean
        self._reintroduce_nans = reintroduce_nans

    def _clean(self, dataset: xarray.Dataset) -> xarray.Dataset:
        """Cleans the dataset of NaNs."""
        data_array = dataset[self._var_to_clean]
        dataset = dataset.assign(
            {self._var_to_clean: data_array.fillna(
                self._fill_value)}
        )
        return dataset

    def _maybe_reintroduce_nans(
        self, stale_inputs: xarray.Dataset, predictions: xarray.Dataset
    ) -> xarray.Dataset:
        # NaN positions don't change between input frames, if they do then
        # we should be more careful about re-introducing them.
        if self._var_to_clean in predictions.keys():
            nan_mask = np.isnan(
                stale_inputs[self._var_to_clean]).any(dim="time")
            with_nan_values = predictions[self._var_to_clean].where(
                ~nan_mask, np.nan)
            predictions = predictions.assign(
                {self._var_to_clean: with_nan_values})
        return predictions

    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> xarray.Dataset:
        if self._reintroduce_nans:
            # Copy inputs before cleaning so that we can reintroduce NaNs later.
            original_inputs = inputs.copy()
        if self._var_to_clean in inputs.keys():
            inputs = self._clean(inputs)
        if forcings and self._var_to_clean in forcings.keys():
            forcings = self._clean(forcings)
        predictions = self._predictor(
            inputs, targets_template, forcings, **kwargs)
        if self._reintroduce_nans:
            predictions = self._maybe_reintroduce_nans(
                original_inputs, predictions)
        return predictions

    def loss(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> base.LossAndDiagnostics:
        if self._var_to_clean in inputs.keys():
            inputs = self._clean(inputs)
        if self._var_to_clean in targets.keys():
            targets = self._clean(targets)
        if forcings and self._var_to_clean in forcings.keys():
            forcings = self._clean(forcings)
        return self._predictor.loss(inputs, targets, forcings, **kwargs)

    def loss_and_predictions(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> Tuple[base.LossAndDiagnostics, xarray.Dataset]:
        if self._reintroduce_nans:
            # Copy inputs before cleaning so that we can reintroduce NaNs later.
            original_inputs = inputs.copy()
        if self._var_to_clean in inputs.keys():
            inputs = self._clean(inputs)
        if self._var_to_clean in targets.keys():
            targets = self._clean(targets)
        if forcings and self._var_to_clean in forcings.keys():
            forcings = self._clean(forcings)

        loss, predictions = self._predictor.loss_and_predictions(
            inputs, targets, forcings, **kwargs
        )
        if self._reintroduce_nans:
            predictions = self._maybe_reintroduce_nans(
                original_inputs, predictions)
        return loss, predictions
