#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/jwohlwend/boltz, 
# licensed under MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro. 

from pathlib import Path

from boltz_data_pipeline.types import Record
from boltz_data_pipeline.filter.dynamic.filter import DynamicFilter


class SubsetFilter(DynamicFilter):
    """Filter a data record based on a subset of the data."""

    def __init__(self, subset: str, reverse: bool = False) -> None:
        """Initialize the filter.

        Parameters
        ----------
        subset : str
            The subset of data to consider, one per line.

        """
        with Path(subset).open("r") as f:
            subset = f.read().splitlines()

        self.subset = {s.lower() for s in subset}
        self.reverse = reverse

    def filter(self, record: Record) -> bool:
        """Filter a data record.

        Parameters
        ----------
        record : Record
            The object to consider filtering in / out.

        Returns
        -------
        bool
            True if the data passes the filter, False otherwise.

        """
        if self.reverse:
            return record.id.lower() not in self.subset
        else:  # noqa: RET505
            return record.id.lower() in self.subset
