#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/jwohlwend/boltz, 
# licensed under MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro. 

from abc import ABC, abstractmethod

import numpy as np

from boltz_data_pipeline.types import Structure


class StaticFilter(ABC):
    """Base class for structure filters."""

    @abstractmethod
    def filter(self, structure: Structure) -> np.ndarray:
        """Filter chains in a structure.

        Parameters
        ----------
        structure : Structure
            The structure to filter chains from.

        Returns
        -------
        np.ndarray
            The chains to keep, as a boolean mask.

        """
        raise NotImplementedError
