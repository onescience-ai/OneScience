

"""Defines protocol for MSA tools."""

import dataclasses
from typing import Protocol


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MsaToolResult:
  """The result of a MSA tool query.

  Attributes:
    target_sequence: The sequence that was used to query the MSA tool.
    e_value: The e-value that was used to filter the MSA tool results.
    a3m: The MSA output of the tool in the A3M format.
    tblout: The optional tblout output of the MSA tool (needed for merging
      results of queries against a sharded database).
  """

  target_sequence: str
  e_value: float
  a3m: str
  tblout: str | None = None


class MsaTool(Protocol):
  """Interface for MSA tools."""

  def query(self, target_sequence: str) -> MsaToolResult:
    """Runs the MSA tool on the target sequence."""
