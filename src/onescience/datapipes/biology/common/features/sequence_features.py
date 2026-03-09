"""Sequence feature extraction module.

This module provides functionality for extracting features from biological sequences,
including amino acid encoding, one-hot encoding, and target feature creation.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from onescience.datapipes.biology.common.features.feature_base import (
    BaseFeatureExtractor,
    FeatureDict,
)
from onescience.datapipes.biology.common.features.constants import (
    SEQUENCE_FEATURE_NAMES,
    RESTYPE_1TO3,
    RESTYPE_3TO1,
    RESTYPES,
    RESTYPE_ORDER,
    RNA_NT_TO_ID,
    DNA_NT_TO_ID,
    STD_RESIDUES,
    STD_RESIDUES_WITH_GAP,
)


class SequenceFeatureExtractor(BaseFeatureExtractor):
    """Sequence feature extractor.

    Extracts features from biological sequences (protein, RNA, DNA),
    including one-hot encoding and residue type features.

    Example:
        >>> extractor = SequenceFeatureExtractor(config={'seq_type': 'protein'})
        >>> features = extractor.extract({'sequence': 'ACDEFGH'})
    """

    def __init__(self, config: Optional[FeatureDict] = None):
        """Initialize the sequence feature extractor.

        Args:
            config: Configuration dictionary containing:
                - seq_type: Sequence type ('protein', 'rna', 'dna').
        """
        super().__init__(config)
        self.seq_type = self.config.get('seq_type', 'protein')

    def extract(self, data: FeatureDict) -> FeatureDict:
        """Extract sequence features from input data.

        Args:
            data: Input data dictionary containing:
                - sequence: Sequence string.

        Returns:
            Dictionary containing sequence features:
                - aatype: Amino acid type indices.
                - target_feat: Target features for structure prediction.
                - seq_length: Sequence length.
        """
        sequence = data.get('sequence', '')

        if not sequence:
            return {}

        features = {}

        # Encode sequence
        aatype = encode_sequence(sequence, self.seq_type)
        features['aatype'] = aatype
        features['seq_length'] = np.array(len(sequence), dtype=np.int32)

        # Create target features
        target_feat = create_target_feat(aatype)
        features['target_feat'] = target_feat

        # One-hot encoding
        onehot = restype_onehot_encode(aatype)
        features['sequence_onehot'] = onehot

        return features


def encode_sequence(
    sequence: str,
    seq_type: str = 'protein'
) -> np.ndarray:
    """Encode sequence string to integer indices.

    Args:
        sequence: Sequence string.
        seq_type: Sequence type ('protein', 'rna', 'dna').

    Returns:
        Array of integer indices.

    Raises:
        ValueError: If seq_type is not supported.
    """
    if seq_type == 'protein':
        mapping = RESTYPES
    elif seq_type == 'rna':
        mapping = RNA_NT_TO_ID
    elif seq_type == 'dna':
        mapping = DNA_NT_TO_ID
    else:
        raise ValueError(f"Unknown sequence type: {seq_type}")

    return np.array([mapping.get(char.upper(), 0) for char in sequence], dtype=np.int32)


def make_sequence_features(
    sequence: str,
    description: Optional[str] = None
) -> FeatureDict:
    """Create sequence features from sequence string.

    Args:
        sequence: Sequence string.
        description: Optional description of the sequence.

    Returns:
        Dictionary containing sequence features.
    """
    seq_length = len(sequence)

    # Encode sequence
    aatype = encode_sequence(sequence, 'protein')

    # Create features
    features = {
        'aatype': aatype,
        'sequence': sequence,
        'seq_length': np.array(seq_length, dtype=np.int32),
        'seq_mask': np.ones(seq_length, dtype=np.float32),
    }

    if description:
        features['description'] = description

    # Add target features
    features['target_feat'] = create_target_feat(aatype)

    return features


def restype_onehot_encode(
    aatype: np.ndarray,
    num_classes: int = 22
) -> np.ndarray:
    """Convert amino acid type indices to one-hot encoding.

    Args:
        aatype: Array of amino acid type indices.
        num_classes: Number of amino acid classes (default: 22 for 20 standard + unknown + gap).

    Returns:
        One-hot encoded array of shape aatype.shape + (num_classes,).
    """
    onehot = np.zeros(aatype.shape + (num_classes,), dtype=np.float32)
    onehot.reshape(-1, num_classes)[np.arange(aatype.size), aatype.reshape(-1)] = 1.0
    return onehot


def create_target_feat(
    aatype: np.ndarray,
    all_atom_positions: Optional[np.ndarray] = None
) -> np.ndarray:
    """Create target features for structure prediction.

    Args:
        aatype: Amino acid type indices of shape (seq_length,).
        all_atom_positions: Optional atom positions of shape (seq_length, num_atoms, 3).

    Returns:
        Target features array.
    """
    seq_length = len(aatype)

    # One-hot encoding of residue type
    target_feat = restype_onehot_encode(aatype, num_classes=21)  # 20 + gap

    return target_feat


def decode_sequence(
    aatype: np.ndarray,
    seq_type: str = 'protein'
) -> str:
    """Decode integer indices to sequence string.

    Args:
        aatype: Array of integer indices.
        seq_type: Sequence type ('protein', 'rna', 'dna').

    Returns:
        Decoded sequence string.
    """
    if seq_type == 'protein':
        mapping = RESTYPE_ORDER
    elif seq_type == 'rna':
        from onescience.datapipes.biology.common.features.constants import RNA_ID_TO_NT
        mapping = RNA_ID_TO_NT
    elif seq_type == 'dna':
        from onescience.datapipes.biology.common.features.constants import DNA_ID_TO_NT
        mapping = DNA_ID_TO_NT
    else:
        raise ValueError(f"Unknown sequence type: {seq_type}")

    return ''.join(mapping[i] for i in aatype if i < len(mapping))


def get_sequence_length(
    features: FeatureDict
) -> int:
    """Get sequence length from features dictionary.

    Args:
        features: Features dictionary containing 'aatype' or 'seq_length'.

    Returns:
        Sequence length.
    """
    if 'seq_length' in features:
        return int(features['seq_length'])
    elif 'aatype' in features:
        return len(features['aatype'])
    else:
        raise ValueError("Features must contain 'aatype' or 'seq_length'")


def validate_sequence(
    sequence: str,
    seq_type: str = 'protein'
) -> Tuple[bool, str]:
    """Validate sequence string.

    Args:
        sequence: Sequence string to validate.
        seq_type: Sequence type ('protein', 'rna', 'dna').

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not sequence:
        return False, "Empty sequence"

    if seq_type == 'protein':
        valid_chars = set(STD_RESIDUES)
    elif seq_type == 'rna':
        valid_chars = set('ACGU')
    elif seq_type == 'dna':
        valid_chars = set('ACGT')
    else:
        return False, f"Unknown sequence type: {seq_type}"

    invalid_chars = set(sequence.upper()) - valid_chars
    if invalid_chars:
        return False, f"Invalid characters: {invalid_chars}"

    return True, ""


def reverse_sequence(
    sequence: Union[str, np.ndarray],
    seq_type: str = 'protein'
) -> Union[str, np.ndarray]:
    """Reverse a sequence.

    Args:
        sequence: Sequence string or array of indices.
        seq_type: Sequence type (only used if sequence is array).

    Returns:
        Reversed sequence.
    """
    if isinstance(sequence, str):
        return sequence[::-1]
    else:
        return sequence[::-1]


def complement_sequence(
    sequence: Union[str, np.ndarray],
    seq_type: str = 'dna'
) -> Union[str, np.ndarray]:
    """Get complement of a nucleotide sequence.

    Args:
        sequence: Sequence string or array of indices.
        seq_type: Sequence type ('dna' or 'rna').

    Returns:
        Complement sequence.
    """
    if seq_type == 'dna':
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
                         'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    elif seq_type == 'rna':
        complement_map = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C',
                         'a': 'u', 'u': 'a', 'c': 'g', 'g': 'c'}
    else:
        raise ValueError(f"Complement not supported for {seq_type}")

    if isinstance(sequence, str):
        return ''.join(complement_map.get(c, c) for c in sequence)
    else:
        # For arrays, convert to string first
        if seq_type == 'dna':
            from onescience.datapipes.biology.common.features.constants import DNA_ID_TO_NT
            mapping = DNA_ID_TO_NT
            to_id = DNA_NT_TO_ID
        else:
            from onescience.datapipes.biology.common.features.constants import RNA_ID_TO_NT
            mapping = RNA_ID_TO_NT
            to_id = RNA_NT_TO_ID

        seq_str = ''.join(mapping[i] for i in sequence if i < len(mapping))
        comp_str = ''.join(complement_map.get(c, c) for c in seq_str)
        return np.array([to_id.get(c.upper(), 0) for c in comp_str], dtype=np.int32)


def reverse_complement(
    sequence: Union[str, np.ndarray],
    seq_type: str = 'dna'
) -> Union[str, np.ndarray]:
    """Get reverse complement of a nucleotide sequence.

    Args:
        sequence: Sequence string or array of indices.
        seq_type: Sequence type ('dna' or 'rna').

    Returns:
        Reverse complement sequence.
    """
    return complement_sequence(reverse_sequence(sequence, seq_type), seq_type)
