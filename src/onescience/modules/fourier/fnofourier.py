import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class FNOFourier(nn.Module):
    """Fourier layer used in the Fourier feature network"""

    def __init__(
        self,
        in_features: int,
        frequencies,
    ) -> None:
        super().__init__()

        # To do: Need more robust way for these params
        if isinstance(frequencies[0], str):
            if "gaussian" in frequencies[0]:
                nr_freq = frequencies[2]
                np_f = (
                    np.random.normal(0, 1, size=(nr_freq, in_features)) * frequencies[1]
                )
            else:
                nr_freq = len(frequencies[1])
                np_f = []
                if "full" in frequencies[0]:
                    np_f_i = np.meshgrid(
                        *[np.array(frequencies[1]) for _ in range(in_features)],
                        indexing="ij",
                    )
                    np_f.append(
                        np.reshape(
                            np.stack(np_f_i, axis=-1),
                            (nr_freq**in_features, in_features),
                        )
                    )
                if "axis" in frequencies[0]:
                    np_f_i = np.zeros((nr_freq, in_features, in_features))
                    for i in range(in_features):
                        np_f_i[:, i, i] = np.reshape(
                            np.array(frequencies[1]), (nr_freq)
                        )
                    np_f.append(
                        np.reshape(np_f_i, (nr_freq * in_features, in_features))
                    )
                if "diagonal" in frequencies[0]:
                    np_f_i = np.reshape(np.array(frequencies[1]), (nr_freq, 1, 1))
                    np_f_i = np.tile(np_f_i, (1, in_features, in_features))
                    np_f_i = np.reshape(np_f_i, (nr_freq * in_features, in_features))
                    np_f.append(np_f_i)
                np_f = np.concatenate(np_f, axis=-2)

        else:
            np_f = frequencies  # [nr_freq, in_features]

        frequencies = torch.tensor(np_f, dtype=torch.get_default_dtype())
        frequencies = frequencies.t().contiguous()
        self.register_buffer("frequencies", frequencies)

    def out_features(self) -> int:
        return int(self.frequencies.size(1) * 2)

    def forward(self, x: Tensor) -> Tensor:
        x_hat = torch.matmul(x, self.frequencies)
        x_sin = torch.sin(2.0 * math.pi * x_hat)
        x_cos = torch.cos(2.0 * math.pi * x_hat)
        x_i = torch.cat([x_sin, x_cos], dim=-1)
        return x_i
    
