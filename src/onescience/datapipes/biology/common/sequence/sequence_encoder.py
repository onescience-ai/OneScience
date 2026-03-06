"""Sequence encoder.

Unified sequence encoding interface.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class SequenceEncoder(ABC):
    """Base class for sequence encoders."""

    @abstractmethod
    def encode(self, sequence: str) -> np.ndarray:
        """Encode a sequence into a numeric array.

        Args:
            sequence: The sequence string.

        Returns:
            The encoded array.
        """
        pass

    @abstractmethod
    def decode(self, encoded: np.ndarray) -> str:
        """Decode an encoded array back to a sequence string.

        Args:
            encoded: The encoded array.

        Returns:
            The sequence string.
        """
        pass


class AminoAcidEncoder(SequenceEncoder):
    """Amino acid sequence encoder.

    Supports standard 20 amino acids plus special characters.
    """

    # Standard 20 amino acids
    STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

    # Extended character mapping
    AA_TO_ID = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        # Special characters
        'X': 20,  # Unknown
        'B': 20,  # Asn or Asp
        'Z': 20,  # Gln or Glu
        'J': 20,  # Leu or Ile
        'U': 20,  # Selenocysteine
        'O': 20,  # Pyrrolysine
        '-': 21,  # Gap
    }

    ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}

    def __init__(self, include_special: bool = True):
        """Initialize the encoder.

        Args:
            include_special: Whether to include special characters (X, B, Z, etc.).
        """
        self.include_special = include_special
        self.vocab_size = 22 if include_special else 20

    def encode(self, sequence: str) -> np.ndarray:
        """Encode an amino acid sequence."""
        encoded = []
        for aa in sequence.upper():
            if aa in self.AA_TO_ID:
                encoded.append(self.AA_TO_ID[aa])
            else:
                # Map unknown characters to X
                encoded.append(self.AA_TO_ID.get('X', 20))
        return np.array(encoded, dtype=np.int32)

    def decode(self, encoded: np.ndarray) -> str:
        """Decode to an amino acid sequence."""
        sequence = []
        for idx in encoded:
            if idx in self.ID_TO_AA:
                sequence.append(self.ID_TO_AA[idx])
            else:
                sequence.append('X')
        return ''.join(sequence)

    def one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode a sequence.

        Args:
            sequence: The sequence string.

        Returns:
            Array of shape (seq_len, vocab_size).
        """
        encoded = self.encode(sequence)
        one_hot = np.zeros((len(encoded), self.vocab_size), dtype=np.float32)
        one_hot[np.arange(len(encoded)), encoded] = 1.0
        return one_hot


class NucleotideEncoder(SequenceEncoder):
    """Nucleotide sequence encoder.

    Supports both DNA and RNA.
    """

    DNA_TO_ID = {
        'A': 0, 'T': 1, 'G': 2, 'C': 3,
        'N': 4,  # Unknown
        '-': 5,  # Gap
    }

    RNA_TO_ID = {
        'A': 0, 'U': 1, 'G': 2, 'C': 3,
        'N': 4,  # Unknown
        '-': 5,  # Gap
    }

    ID_TO_DNA = {v: k for k, v in DNA_TO_ID.items()}
    ID_TO_RNA = {v: k for k, v in RNA_TO_ID.items()}

    def __init__(self, sequence_type: str = "DNA"):
        """Initialize the encoder.

        Args:
            sequence_type: Either "DNA" or "RNA".
        """
        if sequence_type.upper() == "RNA":
            self.to_id = self.RNA_TO_ID
            self.id_to_seq = self.ID_TO_RNA
        else:
            self.to_id = self.DNA_TO_ID
            self.id_to_seq = self.ID_TO_DNA

        self.sequence_type = sequence_type.upper()
        self.vocab_size = 6

    def encode(self, sequence: str) -> np.ndarray:
        """Encode a nucleotide sequence."""
        encoded = []
        for nt in sequence.upper():
            if nt in self.to_id:
                encoded.append(self.to_id[nt])
            else:
                # Map unknown characters to N
                encoded.append(self.to_id.get('N', 4))
        return np.array(encoded, dtype=np.int32)

    def decode(self, encoded: np.ndarray) -> str:
        """Decode to a nucleotide sequence."""
        sequence = []
        for idx in encoded:
            if idx in self.id_to_seq:
                sequence.append(self.id_to_seq[idx])
            else:
                sequence.append('N')
        return ''.join(sequence)

    def one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode a sequence."""
        encoded = self.encode(sequence)
        one_hot = np.zeros((len(encoded), self.vocab_size), dtype=np.float32)
        one_hot[np.arange(len(encoded)), encoded] = 1.0
        return one_hot
