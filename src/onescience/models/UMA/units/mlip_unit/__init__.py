
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch

from onescience.models.UMA.units.mlip_unit.api.inference import (
    InferenceSettings,
    guess_inference_settings,
)
from onescience.models.UMA.units.mlip_unit.predict import MLIPPredictUnit

if TYPE_CHECKING:
    from pathlib import Path


def load_predict_unit(
    path: str | Path,
    inference_settings: InferenceSettings | str = "default",
    overrides: dict | None = None,
    device: Literal["cuda", "cpu"] | None = None,
    atom_refs: dict | None = None,
) -> MLIPPredictUnit:
    """Load a MLIPPredictUnit from a checkpoint file.

    Args:
        path: Path to the checkpoint file
        inference_settings: Settings for inference. Can be "default" (general purpose) or "turbo"
            (optimized for speed but requires fixed atomic composition). Advanced use cases can
            use a custom InferenceSettings object.
        overrides: Optional dictionary of settings to override default inference settings.
        device: Optional torch device to load the model onto.
        atom_refs: Optional dictionary of isolated atom reference energies.

    Returns:
        A MLIPPredictUnit instance ready for inference
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.warning(f"device was not explicitly set, using {device=}.")

    inference_settings = guess_inference_settings(inference_settings)
    overrides = overrides or {"backbone": {"always_use_pbc": False}}

    return MLIPPredictUnit(
        path,
        device=device,
        inference_settings=inference_settings,
        overrides=overrides,
        atom_refs=atom_refs,
    )
