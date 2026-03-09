"""MSA (Multiple Sequence Alignment) feature extraction module.

This module provides functionality for extracting features from MSAs,
including MSA encoding, deletion matrix computation, and row weight calculation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onescience.datapipes.biology.common.features.feature_base import (
    BaseFeatureExtractor,
    FeatureDict,
)
from onescience.datapipes.biology.common.features.constants import (
    MSA_FEATURE_NAMES,
    HHR_PSSM_COLUMNS,
    RESTYPE_ORDER,
    RESTYPES,
    RNA_NT_TO_ID,
    DNA_NT_TO_ID,
)


class MSAFeatureExtractor(BaseFeatureExtractor):
    """MSA feature extractor.

    Extracts features from Multiple Sequence Alignment data, including
    MSA encoding, deletion matrices, and row weights.

    Example:
        >>> extractor = MSAFeatureExtractor(config={'max_msa_clusters': 512})
        >>> features = extractor.extract({'msa': msa_data, 'deletion_matrix': del_matrix})
    """

    def __init__(self, config: Optional[FeatureDict] = None):
        """Initialize the MSA feature extractor.

        Args:
            config: Configuration dictionary containing:
                - max_msa_clusters: Maximum number of MSA sequences to use.
                - max_extra_msa: Maximum number of extra MSA sequences.
        """
        super().__init__(config)
        self.max_msa_clusters = self.config.get('max_msa_clusters', 512)
        self.max_extra_msa = self.config.get('max_extra_msa', 1024)

    def extract(self, data: FeatureDict) -> FeatureDict:
        """Extract MSA features from input data.

        Args:
            data: Input data dictionary containing:
                - msa: MSA array of shape (num_seq, seq_length).
                - deletion_matrix: Deletion matrix of shape (num_seq, seq_length).

        Returns:
            Dictionary containing MSA features:
                - msa_feat: MSA features of shape (num_seq, seq_length, 49).
                - msa_mask: MSA mask of shape (num_seq, seq_length).
                - row_weights: MSA row weights of shape (num_seq,).
        """
        msa = data.get('msa')
        deletion_matrix = data.get('deletion_matrix')

        if msa is None:
            return {}

        features = {}

        # Create MSA features
        msa_feat = create_msa_feat(msa, deletion_matrix)
        features['msa_feat'] = msa_feat

        # Create MSA mask
        msa_mask = make_msa_mask(msa)
        features['msa_mask'] = msa_mask

        # Compute row weights
        row_weights = compute_row_weights(msa)
        features['row_weights'] = row_weights

        return features


def make_msa_features(
    msa_data: List[Tuple[str, str]],
    query_sequence: str
) -> FeatureDict:
    """Create MSA features from MSA data.

    Args:
        msa_data: List of (name, sequence) tuples from MSA file.
        query_sequence: Query sequence string.

    Returns:
        Dictionary containing MSA features.
    """
    if not msa_data:
        # Return empty MSA features
        seq_length = len(query_sequence)
        return {
            'msa': np.array([[RESTYPES.get(aa, 20) for aa in query_sequence]], dtype=np.int32),
            'deletion_matrix': np.zeros((1, seq_length), dtype=np.int32),
            'num_alignments': np.array(1, dtype=np.int32),
        }

    # Parse MSA sequences
    msa_sequences = []
    deletion_matrix = []

    for name, sequence in msa_data:
        # Convert sequence to indices
        indices = []
        deletions = []
        deletion_count = 0

        for char in sequence:
            if char == '-':
                deletion_count += 1
            else:
                indices.append(RESTYPES.get(char.upper(), 20))
                deletions.append(deletion_count)
                deletion_count = 0

        msa_sequences.append(indices)
        deletion_matrix.append(deletions)

    # Convert to arrays
    max_len = max(len(seq) for seq in msa_sequences)
    num_seq = len(msa_sequences)

    msa_array = np.zeros((num_seq, max_len), dtype=np.int32)
    del_array = np.zeros((num_seq, max_len), dtype=np.int32)

    for i, (seq, dels) in enumerate(zip(msa_sequences, deletion_matrix)):
        msa_array[i, :len(seq)] = seq
        del_array[i, :len(dels)] = dels

    return {
        'msa': msa_array,
        'deletion_matrix': del_array,
        'num_alignments': np.array(num_seq, dtype=np.int32),
    }


def make_msa_mask(msa: np.ndarray) -> np.ndarray:
    """Create mask for MSA sequences.

    Args:
        msa: MSA array of shape (num_seq, seq_length).

    Returns:
        Mask array of shape (num_seq, seq_length) where valid positions are 1.
    """
    # Mask is 1 where MSA is not padding (assumes padding is value 0 or gap)
    return (msa != 0).astype(np.float32)


def create_msa_feat(
    msa: np.ndarray,
    deletion_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """Create MSA features from MSA and deletion matrix.

    Args:
        msa: MSA array of shape (num_seq, seq_length).
        deletion_matrix: Deletion matrix of shape (num_seq, seq_length).
            If None, assumes no deletions.

    Returns:
        MSA features of shape (num_seq, seq_length, 49).
            - Dimension 0-20: One-hot encoding of amino acid type.
            - Dimension 21: Has deletion.
            - Dimension 22-48: Deletion value (binned).
    """
    num_seq, seq_length = msa.shape

    # One-hot encode MSA
    msa_onehot = np.zeros((num_seq, seq_length, 22), dtype=np.float32)
    msa_onehot[np.arange(num_seq)[:, None], np.arange(seq_length), msa] = 1.0

    # Handle deletion matrix
    if deletion_matrix is None:
        deletion_matrix = np.zeros((num_seq, seq_length), dtype=np.int32)

    has_deletion = (deletion_matrix > 0).astype(np.float32)

    # Bin deletion values (0-30)
    deletion_value = np.clip(deletion_matrix, 0, 30).astype(np.float32)
    deletion_bins = np.zeros((num_seq, seq_length, 27), dtype=np.float32)
    deletion_bins[np.arange(num_seq)[:, None], np.arange(seq_length), deletion_matrix.clip(0, 26)] = 1.0

    # Concatenate features
    msa_feat = np.concatenate([
        msa_onehot[:, :, :21],  # 20 amino acids + gap
        has_deletion[:, :, None],
        deletion_bins,
    ], axis=-1)

    return msa_feat


def create_deletion_matrix(
    msa_sequences: List[str],
    query_sequence: str
) -> np.ndarray:
    """Create deletion matrix from MSA sequences.

    Args:
        msa_sequences: List of MSA sequence strings.
        query_sequence: Query sequence for alignment reference.

    Returns:
        Deletion matrix of shape (num_seq, seq_length).
    """
    num_seq = len(msa_sequences)
    seq_length = len(query_sequence)
    deletion_matrix = np.zeros((num_seq, seq_length), dtype=np.int32)

    for i, seq in enumerate(msa_sequences):
        deletion_count = 0
        position = 0

        for char in seq:
            if char == '-':
                deletion_count += 1
            else:
                if position < seq_length:
                    deletion_matrix[i, position] = deletion_count
                    deletion_count = 0
                    position += 1

    return deletion_matrix


def compute_row_weights(msa: np.ndarray) -> np.ndarray:
    """Compute weights for MSA rows based on sequence similarity.

    Uses the Neff (effective number of sequences) weighting scheme.

    Args:
        msa: MSA array of shape (num_seq, seq_length).

    Returns:
        Row weights array of shape (num_seq,).
    """
    num_seq = msa.shape[0]

    if num_seq == 1:
        return np.ones(1, dtype=np.float32)

    # Compute pairwise sequence identity
    weights = np.zeros(num_seq, dtype=np.float32)

    for i in range(num_seq):
        # Count sequences that are >80% identical
        identical_count = 0
        for j in range(num_seq):
            if i == j:
                continue
            # Compute identity (excluding gaps)
            valid_mask = (msa[i] != 0) & (msa[j] != 0)
            if valid_mask.sum() == 0:
                continue
            identity = (msa[i][valid_mask] == msa[j][valid_mask]).mean()
            if identity > 0.8:
                identical_count += 1

        # Weight is inverse of cluster size
        weights[i] = 1.0 / (1 + identical_count)

    # Normalize weights
    weights = weights / weights.sum() * num_seq

    return weights.astype(np.float32)


def cluster_msa(
    msa: np.ndarray,
    num_clusters: int,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster MSA sequences using sequence similarity.

    Args:
        msa: MSA array of shape (num_seq, seq_length).
        num_clusters: Number of clusters to create.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (cluster_centers, cluster_assignments) where:
            - cluster_centers: Indices of cluster center sequences.
            - cluster_assignments: Cluster assignment for each sequence.
    """
    np.random.seed(seed)
    num_seq = msa.shape[0]

    if num_seq <= num_clusters:
        return np.arange(num_seq), np.arange(num_seq)

    # Randomly select cluster centers (keeping first sequence as center)
    cluster_centers = np.concatenate([
        np.array([0]),
        np.random.choice(num_seq - 1, num_clusters - 1, replace=False) + 1
    ])

    # Assign each sequence to nearest center
    assignments = np.zeros(num_seq, dtype=np.int32)

    for i in range(num_seq):
        min_dist = float('inf')
        for j, center_idx in enumerate(cluster_centers):
            # Compute Hamming distance (excluding gaps)
            valid_mask = (msa[i] != 0) & (msa[center_idx] != 0)
            if valid_mask.sum() == 0:
                dist = seq_length
            else:
                dist = (msa[i][valid_mask] != msa[center_idx][valid_mask]).sum()

            if dist < min_dist:
                min_dist = dist
                assignments[i] = j

    return cluster_centers, assignments


def parse_a3m(a3m_string: str) -> List[Tuple[str, str]]:
    """Parse A3M format MSA file.

    Args:
        a3m_string: Contents of A3M file as string.

    Returns:
        List of (name, sequence) tuples.
    """
    sequences = []
    current_name = None
    current_seq = []

    for line in a3m_string.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('>'):
            # Save previous sequence
            if current_name is not None:
                sequences.append((current_name, ''.join(current_seq)))
            current_name = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)

    # Save last sequence
    if current_name is not None:
        sequences.append((current_name, ''.join(current_seq)))

    return sequences


def parse_stockholm(stockholm_string: str) -> List[Tuple[str, str]]:
    """Parse Stockholm format MSA file.

    Args:
        stockholm_string: Contents of Stockholm file as string.

    Returns:
        List of (name, sequence) tuples.
    """
    sequences = {}

    for line in stockholm_string.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue

        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            seq = parts[1]
            if name not in sequences:
                sequences[name] = []
            sequences[name].append(seq)

    return [(name, ''.join(seqs)) for name, seqs in sequences.items()]
