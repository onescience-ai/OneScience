"""Unified MSA parser.

Supports a3m and Stockholm formats.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence
from pathlib import Path
import re

from onescience.datapipes.biology.common.utils.file_utils import (
    open_file,
    detect_file_format,
)


@dataclass
class MSA:
    """Unified MSA data format.

    Attributes:
        sequences: List of sequences.
        deletion_matrix: Deletion matrix.
        descriptions: List of sequence descriptions.
    """
    sequences: List[str]
    deletion_matrix: List[List[int]]
    descriptions: List[str]
    
    def __post_init__(self):
        """Validates data consistency."""
        if not (len(self.sequences) == len(self.deletion_matrix) == len(self.descriptions)):
            raise ValueError(
                "All fields for an MSA must have the same length. "
                f"Got sequences={len(self.sequences)}, "
                f"deletion_matrix={len(self.deletion_matrix)}, "
                f"descriptions={len(self.descriptions)}"
            )
    
    def __len__(self) -> int:
        """Returns the number of sequences."""
        return len(self.sequences)
    
    def truncate(self, max_seqs: int) -> "MSA":
        """Truncates the MSA to the specified number of sequences.

        Args:
            max_seqs: Maximum number of sequences.

        Returns:
            Truncated MSA.
        """
        return MSA(
            sequences=self.sequences[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            descriptions=self.descriptions[:max_seqs],
        )


class MSAParser:
    """Unified MSA parser.

    Supported formats:
    - a3m (HH-suite format)
    - Stockholm (Stockholm format)
    """
    
    @staticmethod
    def parse_a3m(a3m_string: str) -> MSA:
        """Parses an MSA in a3m format.

        Args:
            a3m_string: String in a3m format.

        Returns:
            Parsed MSA object.
        """
        sequences = []
        deletion_matrix = []
        descriptions = []

        current_seq = ""
        current_deletions = []
        current_desc = ""

        for line in a3m_string.splitlines():
            line = line.strip()

            if line.startswith(">"):
                # Save the previous sequence
                if current_seq:
                    sequences.append(current_seq)
                    deletion_matrix.append(current_deletions)
                    descriptions.append(current_desc)

                # Start a new sequence
                current_desc = line[1:]  # Remove '>'
                current_seq = ""
                current_deletions = []
            elif line:
                # Parse sequence line (lowercase letters indicate insertions)
                seq_line = ""
                del_line = []

                for char in line:
                    if char.isupper() or char == '-':
                        # Standard character or gap
                        seq_line += char
                        del_line.append(0)
                    elif char.islower():
                        # Lowercase letters indicate insertions, kept in sequence
                        seq_line += char.upper()
                        del_line.append(0)
                    elif char.isdigit():
                        # Digits indicate number of deleted columns
                        if del_line:
                            del_line[-1] = del_line[-1] * 10 + int(char)
                        else:
                            del_line.append(int(char))
                    else:
                        # Skip other characters (e.g., spaces)
                        continue

                current_seq += seq_line
                # Expand deletion matrix
                while len(current_deletions) < len(current_seq):
                    current_deletions.append(0)
                # Update deletion matrix
                for i, del_val in enumerate(del_line):
                    idx = len(current_seq) - len(del_line) + i
                    if idx < len(current_deletions):
                        current_deletions[idx] = del_val

        # Save the last sequence
        if current_seq:
            sequences.append(current_seq)
            deletion_matrix.append(current_deletions)
            descriptions.append(current_desc)

        return MSA(
            sequences=sequences,
            deletion_matrix=deletion_matrix,
            descriptions=descriptions,
        )
    
    @staticmethod
    def parse_stockholm(sto_string: str) -> MSA:
        """Parses an MSA in Stockholm format.

        Args:
            sto_string: String in Stockholm format.

        Returns:
            Parsed MSA object.
        """
        sequences = []
        deletion_matrix = []
        descriptions = []

        seq_dict = {}  # name -> sequence
        in_alignment = False

        for line in sto_string.splitlines():
            line = line.strip()

            if line.startswith("# STOCKHOLM"):
                in_alignment = True
                continue
            elif line.startswith("//"):
                in_alignment = False
                break
            elif line.startswith("#"):
                continue  # Skip comments
            elif not line:
                continue  # Skip empty lines
            elif in_alignment and not line.startswith("#"):
                # Parse sequence line
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    sequence = parts[1]

                    if name not in seq_dict:
                        seq_dict[name] = ""
                        descriptions.append(name)

                    seq_dict[name] += sequence

        # Convert to list format
        for desc in descriptions:
            seq = seq_dict[desc]
            sequences.append(seq)

            # Compute deletion matrix ('.' indicates deletion in Stockholm format)
            deletions = []
            for char in seq:
                if char == '.':
                    deletions.append(1)
                else:
                    deletions.append(0)
            deletion_matrix.append(deletions)

        return MSA(
            sequences=sequences,
            deletion_matrix=deletion_matrix,
            descriptions=descriptions,
        )
    
    @staticmethod
    def parse_file(path: Path, format: Optional[str] = None) -> MSA:
        """Parses an MSA from a file (supports compressed files).

        Args:
            path: Path to the MSA file (supports .gz, .bz2, .xz compressed files).
            format: Format ("a3m" or "stockholm"). If None, auto-detects.

        Returns:
            Parsed MSA object.
        """
        with open_file(path, 'r') as f:
            content = f.read()
        
        # 自动检测格式
        if format is None:
            # 首先尝试基于扩展名和内容检测
            detected_format = detect_file_format(path)
            if detected_format in ['a3m', 'stockholm', 'clustal', 'phylip']:
                format = detected_format
            elif '>' in content[:100] and any(c.islower() for c in content[:1000]):
                format = 'a3m'
            elif '# STOCKHOLM' in content:
                format = 'stockholm'
            else:
                raise ValueError(
                    f"Could not detect MSA format for file: {path}. "
                    f"Supported formats: a3m, stockholm, clustal, phylip"
                )
        
        if format == 'a3m':
            return MSAParser.parse_a3m(content)
        elif format == 'stockholm':
            return MSAParser.parse_stockholm(content)
        else:
            raise ValueError(
                f"Unsupported MSA format: {format}. "
                f"Currently only 'a3m' and 'stockholm' are implemented. "
                f"Detected format: {format}"
            )

