#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/jwohlwend/boltz, 
# licensed under MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro. 

from boltz_data_pipeline.types import Record
from boltz_data_pipeline.filter.dynamic.filter import DynamicFilter


class ResolutionFilter(DynamicFilter):
    """A filter that filters complexes based on their resolution."""

    def __init__(self, resolution: float = 9.0) -> None:
        """Initialize the filter.

        Parameters
        ----------
        resolution : float, optional
            The maximum allowed resolution.

        """
        self.resolution = resolution

    def filter(self, record: Record) -> bool:
        """Filter complexes based on their resolution.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        structure = record.structure
        return structure.resolution <= self.resolution
