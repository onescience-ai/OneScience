"""Unified MSA feature extractor.

Extracts features from MSA for use by various models.
"""

from typing import Dict, List, Optional
import numpy as np
from onescience.datapipes.biology.common.msa.msa_parser import MSA
from onescience.datapipes.biology.common.sequence.sequence_encoder import AminoAcidEncoder


class MSAFeaturizer:
    """Unified MSA feature extractor.

    Extracted features include:
    - MSA sequence matrix
    - Deletion matrix
    - Sequence weights
    - Sequence diversity
    """
    
    def __init__(self, max_seqs: Optional[int] = None, encoder: Optional[AminoAcidEncoder] = None):
        """Initializes the MSA featurizer.

        Args:
            max_seqs: Maximum number of sequences. If None, no limit is applied.
            encoder: Sequence encoder. If None, uses the default encoder.
        """
        self.max_seqs = max_seqs
        self.encoder = encoder or AminoAcidEncoder()
    
    def featurize(self, msa: MSA) -> Dict[str, np.ndarray]:
        """Extracts MSA features.

        Args:
            msa: MSA object.

        Returns:
            Dictionary of features.
        """
        # Truncate sequences if needed
        if self.max_seqs and len(msa) > self.max_seqs:
            msa = msa.truncate(self.max_seqs)

        features = {}

        # MSA sequence matrix
        features['msa'] = self._create_msa_matrix(msa)

        # Deletion matrix
        features['deletion_matrix'] = self._create_deletion_matrix(msa)

        # Sequence weights (simple implementation: equal weights)
        features['msa_row_weights'] = self._compute_row_weights(msa)

        # Number of sequences
        features['num_alignments'] = np.array(len(msa), dtype=np.int32)

        return features
    
    def _create_msa_matrix(self, msa: MSA) -> np.ndarray:
        """Creates the MSA sequence matrix.

        Returns:
            Array with shape (num_seqs, seq_len).
        """
        if not msa.sequences:
            return np.array([], dtype=np.int32).reshape(0, 0)

        # Find the maximum sequence length
        max_len = max(len(seq) for seq in msa.sequences)

        # Create matrix using the sequence encoder
        msa_matrix = []
        for seq in msa.sequences:
            # Encode the sequence using the encoder
            encoded = self.encoder.encode(seq)
            # Pad to maximum length using gap encoding
            if len(encoded) < max_len:
                gap_code = self.encoder.AA_TO_ID.get('-', 21)
                padding = np.full(max_len - len(encoded), gap_code, dtype=np.int32)
                encoded = np.concatenate([encoded, padding])
            msa_matrix.append(encoded)

        return np.array(msa_matrix, dtype=np.int32)
    
    def _create_deletion_matrix(self, msa: MSA) -> np.ndarray:
        """Creates the deletion matrix.

        Returns:
            Array with shape (num_seqs, seq_len).
        """
        if not msa.deletion_matrix:
            return np.array([], dtype=np.int32).reshape(0, 0)

        # Find the maximum length
        max_len = max(len(del_row) for del_row in msa.deletion_matrix)

        # Create matrix
        del_matrix = []
        for del_row in msa.deletion_matrix:
            # Pad to maximum length
            if len(del_row) < max_len:
                del_row = list(del_row) + [0] * (max_len - len(del_row))
            del_matrix.append(del_row)

        return np.array(del_matrix, dtype=np.int32)
    
    def _compute_row_weights(self, msa: MSA) -> np.ndarray:
        """Computes sequence weights.

        Simple implementation: equal weights.
        Can be extended to more complex weight calculations (e.g., Henikoff weights).

        Returns:
            Array with shape (num_seqs,).
        """
        num_seqs = len(msa)
        return np.ones(num_seqs, dtype=np.float32) / num_seqs

