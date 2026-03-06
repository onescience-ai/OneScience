"""Unified FASTA parser.

Common functionality extracted from protenix and openfold.
Supports compressed files (.gz, .bz2, .xz).
"""

from typing import List, Tuple, Iterator
from pathlib import Path
import re

from onescience.datapipes.biology.common.utils.file_utils import (
    open_file,
    is_compressed,
    get_base_extension,
)


class FASTAParser:
    """Unified FASTA parser.

    Supports:
    - FASTA string parsing
    - FASTA file parsing
    - Multi-sequence FASTA parsing
    """

    @staticmethod
    def parse(fasta_string: str) -> Tuple[List[str], List[str]]:
        """Parse a FASTA string.

        Args:
            fasta_string: A string in FASTA format.

        Returns:
            A tuple containing a list of sequences and a list of descriptions.
        """
        sequences = []
        descriptions = []
        index = -1

        for line in fasta_string.splitlines():
            line = line.strip()
            if line.startswith(">"):
                index += 1
                descriptions.append(line[1:])  # Remove '>'
                sequences.append("")
            elif line.startswith("#"):
                continue  # Skip comment lines
            elif not line:
                continue  # Skip empty lines
            else:
                sequences[index] += line

        return sequences, descriptions

    @staticmethod
    def parse_file(path: Path) -> Tuple[List[str], List[str]]:
        """Parse a FASTA from file (supports compressed files).

        Args:
            path: Path to the FASTA file (supports .gz, .bz2, .xz compressed files).

        Returns:
            A tuple containing a list of sequences and a list of descriptions.
        """
        with open_file(path, 'r') as f:
            fasta_string = f.read()
        return FASTAParser.parse(fasta_string)

    @staticmethod
    def parse_file_stream(path: Path) -> Iterator[Tuple[str, str]]:
        """Parse a FASTA file in streaming mode (supports large and compressed files).

        Args:
            path: Path to the FASTA file (supports compressed files).

        Yields:
            Tuples of (sequence, description).
        """
        current_seq = ""
        current_desc = ""

        with open_file(path, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith(">"):
                    # Save the previous sequence
                    if current_seq:
                        yield current_seq, current_desc

                    # Start a new sequence
                    current_desc = line[1:]  # Remove '>'
                    current_seq = ""
                elif line.startswith("#"):
                    continue  # Skip comment lines
                elif line:
                    current_seq += line

        # Save the last sequence
        if current_seq:
            yield current_seq, current_desc

    @staticmethod
    def parse_single(fasta_string: str) -> Tuple[str, str]:
        """Parse a single sequence FASTA.

        Args:
            fasta_string: A string in FASTA format (containing only one sequence).

        Returns:
            A tuple containing the sequence and its description.

        Raises:
            ValueError: If no sequence is found or if multiple sequences are found.
        """
        sequences, descriptions = FASTAParser.parse(fasta_string)
        if len(sequences) == 0:
            raise ValueError("No sequence found in FASTA string")
        if len(sequences) > 1:
            raise ValueError("Multiple sequences found, use parse() instead")
        return sequences[0], descriptions[0]
