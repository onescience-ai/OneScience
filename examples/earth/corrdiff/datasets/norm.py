import numpy as np


def normalize(x, center, scale):
    """Normalize input data 'x' using center and scale values."""
    # x.shape=(channels, 450, 450), center.shape=(channels,), scale.shape=(channels,)
    center = np.asarray(center)
    scale = np.asarray(scale)
    if not (center.ndim == 1 and scale.ndim == 1):
        raise ValueError(
            "center and scale must be 1D arrays")
    return (x - center[np.newaxis, :, np.newaxis, np.newaxis]) / scale[
        np.newaxis, :, np.newaxis, np.newaxis
    ]


def denormalize(x, center, scale):
    """Denormalize input data 'x' using center and scale values."""
    center = np.asarray(center)
    scale = np.asarray(scale)
    if not (center.ndim == 1 and scale.ndim == 1):
        raise ValueError(
            "center and scale must be 1D arrays")
    return (
        x * scale[np.newaxis, :, np.newaxis, np.newaxis]
        + center[np.newaxis, :, np.newaxis, np.newaxis]
    )
