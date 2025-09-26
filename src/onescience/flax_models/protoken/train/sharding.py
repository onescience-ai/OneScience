from typing import Union

import jax
import numpy as np

Array = Union[jax.Array, np.ndarray]


def _sharding(input, shards):

    n_device = shards.shape[0]
    if isinstance(input, (np.ndarray, jax.Array)):
        _shape = [
            n_device,
        ] + [1 for _ in range(input.ndim - 1)]
        return jax.device_put(input, shards.reshape(_shape))
    elif input is None:
        return jax.device_put(input, shards)
    else:
        raise TypeError(
            f"Invalid input: {type(input)} {input}")
