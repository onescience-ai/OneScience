"""Token-level feature extraction module.

This module provides functionality for extracting token-level features,
including atom-to-token mapping, reference position encoding, and bond features.
Inspired by the Protenix implementation.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from onescience.datapipes.biology.common.features.feature_base import (
    BaseFeatureExtractor,
    FeatureDict,
)
from onescience.datapipes.biology.common.features.constants import (
    ATOM_TYPES,
    ATOM_ORDER,
    RESTYPE_ORDER,
    RESTYPES,
)


# Token element encoding
elems = [
    "C", "N", "O", "S", "P",
    "SE", "NA", "CL", "K", "CA",
    "MG", "ZN", "FE", "CU", "MN",
    "CO", "F", "I", "BR", "X"
]
elem_to_index = {elem: i for i, elem in enumerate(elems)}


def encoder(
    value: Union[str, int, float],
    encoding_map: Dict,
    default: int = 0
) -> int:
    """Encode a value using a mapping dictionary.

    Args:
        value: Value to encode.
        encoding_map: Dictionary mapping values to indices.
        default: Default value if not found in mapping.

    Returns:
        Encoded index.
    """
    return encoding_map.get(value, default)


def restype_onehot_encoded(
    restype: Union[str, int],
    num_classes: int = 32
) -> np.ndarray:
    """Get one-hot encoding for residue type.

    Args:
        restype: Residue type (single letter code or index).
        num_classes: Number of residue classes.

    Returns:
        One-hot encoded array.
    """
    if isinstance(restype, str):
        restype_idx = RESTYPES.get(restype.upper(), num_classes - 1)
    else:
        restype_idx = restype

    onehot = np.zeros(num_classes, dtype=np.float32)
    if 0 <= restype_idx < num_classes:
        onehot[restype_idx] = 1.0
    return onehot


def elem_onehot_encoded(
    elem: str,
    num_classes: int = 20
) -> np.ndarray:
    """Get one-hot encoding for element type.

    Args:
        elem: Element symbol (e.g., 'C', 'N', 'O').
        num_classes: Number of element classes.

    Returns:
        One-hot encoded array.
    """
    elem_idx = elem_to_index.get(elem.upper(), num_classes - 1)

    onehot = np.zeros(num_classes, dtype=np.float32)
    if 0 <= elem_idx < num_classes:
        onehot[elem_idx] = 1.0
    return onehot


def ref_atom_name_chars_encoded(
    atom_name: str,
    max_len: int = 4
) -> np.ndarray:
    """Encode atom name as character indices.

    Args:
        atom_name: Atom name string (e.g., 'CA', 'N', 'O').
        max_len: Maximum length of atom name.

    Returns:
        Array of character indices.
    """
    encoded = np.zeros(max_len, dtype=np.int32)
    for i, char in enumerate(atom_name[:max_len].upper()):
        if char.isalpha():
            encoded[i] = ord(char) - ord('A') + 1  # A=1, B=2, etc.
        elif char.isdigit():
            encoded[i] = ord(char) - ord('0') + 27  # 0=27, 1=28, etc.
    return encoded


def get_prot_nuc_frame_atom_names() -> Dict[str, List[str]]:
    """Get frame atom names for protein and nucleic acid residues.

    Returns:
        Dictionary mapping residue names to frame atom names.
    """
    return {
        # Protein backbone frame atoms
        'ALA': ['N', 'CA', 'C'],
        'CYS': ['N', 'CA', 'C'],
        'ASP': ['N', 'CA', 'C'],
        'GLU': ['N', 'CA', 'C'],
        'PHE': ['N', 'CA', 'C'],
        'GLY': ['N', 'CA', 'C'],
        'HIS': ['N', 'CA', 'C'],
        'ILE': ['N', 'CA', 'C'],
        'LYS': ['N', 'CA', 'C'],
        'LEU': ['N', 'CA', 'C'],
        'MET': ['N', 'CA', 'C'],
        'ASN': ['N', 'CA', 'C'],
        'PRO': ['N', 'CA', 'C'],
        'GLN': ['N', 'CA', 'C'],
        'ARG': ['N', 'CA', 'C'],
        'SER': ['N', 'CA', 'C'],
        'THR': ['N', 'CA', 'C'],
        'VAL': ['N', 'CA', 'C'],
        'TRP': ['N', 'CA', 'C'],
        'TYR': ['N', 'CA', 'C'],
        # DNA backbone frame atoms
        'DA': ["P", "O5'", "C5'"],
        'DC': ["P", "O5'", "C5'"],
        'DG': ["P", "O5'", "C5'"],
        'DT': ["P", "O5'", "C5'"],
        # RNA backbone frame atoms
        'A': ["P", "O5'", "C5'"],
        'C': ["P", "O5'", "C5'"],
        'G': ["P", "O5'", "C5'"],
        'U': ["P", "O5'", "C5'"],
    }


def check_colinear(
    positions: np.ndarray,
    threshold: float = 0.9
) -> bool:
    """Check if three points are colinear.

    Args:
        positions: Array of 3 positions, shape (3, 3).
        threshold: Cosine similarity threshold for colinearity.

    Returns:
        True if points are colinear.
    """
    if positions.shape != (3, 3):
        return False

    v1 = positions[1] - positions[0]
    v2 = positions[2] - positions[1]

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return True

    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    return abs(cos_angle) > threshold


def compute_frame_from_positions(
    positions: np.ndarray
) -> np.ndarray:
    """Compute a local frame from 3 positions.

    Args:
        positions: Array of 3 positions, shape (3, 3).

    Returns:
        4x4 transformation matrix (rotation + translation).
    """
    if check_colinear(positions):
        # Return identity-like frame for colinear points
        frame = np.eye(4, dtype=np.float32)
        frame[:3, 3] = positions[1]  # Use middle point as origin
        return frame

    # Origin at second point
    origin = positions[1]

    # X-axis: from point 1 to point 2
    x_axis = positions[2] - positions[1]
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Z-axis: perpendicular to plane
    v1 = positions[1] - positions[0]
    z_axis = np.cross(v1, x_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Y-axis: complete right-handed system
    y_axis = np.cross(z_axis, x_axis)

    # Construct frame matrix
    frame = np.eye(4, dtype=np.float32)
    frame[:3, 0] = x_axis
    frame[:3, 1] = y_axis
    frame[:3, 2] = z_axis
    frame[:3, 3] = origin

    return frame


def get_token_features_from_annotations(
    atom_positions: np.ndarray,
    atom_names: List[str],
    residue_names: List[str],
    chain_ids: List[str]
) -> FeatureDict:
    """Extract token features from atom-level annotations.

    Args:
        atom_positions: Atom coordinates of shape (num_atoms, 3).
        atom_names: List of atom names.
        residue_names: List of residue names.
        chain_ids: List of chain IDs.

    Returns:
        Dictionary containing token features.
    """
    num_atoms = len(atom_positions)

    features = {
        'atom_positions': atom_positions,
        'atom_names': atom_names,
        'residue_names': residue_names,
        'chain_ids': chain_ids,
        'num_atoms': num_atoms,
    }

    return features


def get_reference_features(
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    reference_positions: np.ndarray
) -> np.ndarray:
    """Compute reference position features.

    Args:
        atom_positions: Atom coordinates of shape (num_atoms, 3).
        atom_mask: Atom mask of shape (num_atoms,).
        reference_positions: Reference positions of shape (num_refs, 3).

    Returns:
        Reference features of shape (num_atoms, num_refs).
    """
    # Compute distances to reference positions
    diff = atom_positions[:, None, :] - reference_positions[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Apply mask
    distances = distances * atom_mask[:, None]

    return distances


def get_bond_features(
    atom_positions: np.ndarray,
    bonds: List[Tuple[int, int]]
) -> np.ndarray:
    """Compute bond features from atom positions and bond list.

    Args:
        atom_positions: Atom coordinates of shape (num_atoms, 3).
        bonds: List of (atom_i, atom_j) tuples defining bonds.

    Returns:
        Bond features including distances.
    """
    bond_features = []

    for i, j in bonds:
        dist = np.linalg.norm(atom_positions[i] - atom_positions[j])
        bond_features.append(dist)

    return np.array(bond_features, dtype=np.float32)


def classify_polymer_bonds(
    residue_names: List[str],
    atom_names: List[str]
) -> List[Tuple[int, int]]:
    """Classify bonds within polymer chains.

    Args:
        residue_names: List of residue names.
        atom_names: List of atom names.

    Returns:
        List of (atom_i, atom_j) tuples for bonds.
    """
    bonds = []

    # Simple peptide bond detection
    for i in range(len(residue_names) - 1):
        # C(i) - N(i+1) peptide bond
        # This is a simplified version; real implementation would need proper atom indexing
        pass

    return bonds


def get_chain_perm_features(
    chain_ids: List[str],
    num_chains: int
) -> np.ndarray:
    """Get chain permutation features.

    Args:
        chain_ids: List of chain IDs for each atom/token.
        num_chains: Total number of chains.

    Returns:
        Chain permutation encoding.
    """
    # Create chain ID to index mapping
    unique_chains = sorted(set(chain_ids))
    chain_to_idx = {c: i for i, c in enumerate(unique_chains)}

    # Encode chain IDs
    chain_indices = [chain_to_idx[c] for c in chain_ids]

    return np.array(chain_indices, dtype=np.int32)


def get_extra_features(
    atom_positions: np.ndarray,
    atom_features: np.ndarray,
    cutoff: float = 5.0
) -> np.ndarray:
    """Compute extra features based on local neighborhood.

    Args:
        atom_positions: Atom coordinates of shape (num_atoms, 3).
        atom_features: Atom features of shape (num_atoms, feature_dim).
        cutoff: Distance cutoff for neighborhood.

    Returns:
        Aggregated extra features.
    """
    num_atoms = len(atom_positions)

    # Compute pairwise distances
    diff = atom_positions[:, None, :] - atom_positions[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Find neighbors within cutoff
    neighbors = distances < cutoff

    # Aggregate features from neighbors
    extra_features = []
    for i in range(num_atoms):
        neighbor_mask = neighbors[i]
        if neighbor_mask.sum() > 0:
            neighbor_features = atom_features[neighbor_mask].mean(axis=0)
        else:
            neighbor_features = np.zeros(atom_features.shape[1], dtype=np.float32)
        extra_features.append(neighbor_features)

    return np.array(extra_features, dtype=np.float32)


def get_mask_features(
    atom_mask: np.ndarray,
    residue_mask: Optional[np.ndarray] = None
) -> FeatureDict:
    """Get mask features for atoms and residues.

    Args:
        atom_mask: Atom mask of shape (num_atoms,).
        residue_mask: Optional residue mask.

    Returns:
        Dictionary containing mask features.
    """
    features = {
        'atom_mask': atom_mask.astype(np.float32),
    }

    if residue_mask is not None:
        features['residue_mask'] = residue_mask.astype(np.float32)

    return features


def get_label_features(
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    resolution: float = 1.0
) -> FeatureDict:
    """Get label features for training.

    Args:
        atom_positions: Atom coordinates of shape (num_atoms, 3).
        atom_mask: Atom mask of shape (num_atoms,).
        resolution: Structure resolution.

    Returns:
        Dictionary containing label features.
    """
    features = {
        'atom_positions': atom_positions,
        'atom_mask': atom_mask.astype(np.float32),
        'resolution': np.array(resolution, dtype=np.float32),
    }

    return features


class TokenFeatureExtractor(BaseFeatureExtractor):
    """Token-level feature extractor.

    Extracts token-level features for atom-based representations,
    including atom type encoding, position features, and bond information.

    Example:
        >>> extractor = TokenFeatureExtractor(config={'use_bonds': True})
        >>> features = extractor.extract({
        ...     'atom_positions': positions,
        ...     'atom_names': names,
        ...     'residue_names': residues
        ... })
    """

    def __init__(self, config: Optional[FeatureDict] = None):
        """Initialize the token feature extractor.

        Args:
            config: Configuration dictionary containing:
                - use_bonds: Whether to compute bond features.
                - use_frames: Whether to compute frame features.
        """
        super().__init__(config)
        self.use_bonds = self.config.get('use_bonds', True)
        self.use_frames = self.config.get('use_frames', True)

    def extract(self, data: FeatureDict) -> FeatureDict:
        """Extract token features from input data.

        Args:
            data: Input data dictionary containing:
                - atom_positions: Atom coordinates.
                - atom_names: Atom names.
                - residue_names: Residue names.
                - chain_ids: Chain IDs.

        Returns:
            Dictionary containing token features.
        """
        atom_positions = data.get('atom_positions')
        atom_names = data.get('atom_names', [])
        residue_names = data.get('residue_names', [])
        chain_ids = data.get('chain_ids', [])

        if atom_positions is None:
            return {}

        features = {}

        # Basic token features
        features.update(get_token_features_from_annotations(
            atom_positions, atom_names, residue_names, chain_ids
        ))

        # Atom type encoding
        atom_type_features = []
        for name in atom_names:
            elem = name[0] if name else 'C'
            atom_type_features.append(elem_onehot_encoded(elem))
        features['atom_type_onehot'] = np.array(atom_type_features, dtype=np.float32)

        # Residue type encoding
        residue_features = []
        for res in residue_names:
            residue_features.append(restype_onehot_encoded(res[:3] if len(res) >= 3 else res))
        features['residue_type_onehot'] = np.array(residue_features, dtype=np.float32)

        return features


def create_atom_to_token_mapping(
    atom_names: List[str],
    residue_names: List[str],
    tokens_per_residue: int = 1
) -> np.ndarray:
    """Create mapping from atoms to tokens.

    Args:
        atom_names: List of atom names.
        residue_names: List of residue names.
        tokens_per_residue: Number of tokens per residue.

    Returns:
        Array mapping each atom to a token index.
    """
    num_atoms = len(atom_names)
    mapping = np.zeros(num_atoms, dtype=np.int32)

    # Simple mapping: group atoms by residue
    current_residue = None
    token_idx = 0

    for i, res in enumerate(residue_names):
        if res != current_residue:
            current_residue = res
            token_idx += 1
        mapping[i] = token_idx - 1

    return mapping


def validate_frame_atoms(
    atom_positions: np.ndarray,
    frame_atom_indices: List[int]
) -> bool:
    """Validate that frame atoms form a valid frame.

    Args:
        atom_positions: Atom coordinates.
        frame_atom_indices: Indices of 3 atoms defining the frame.

    Returns:
        True if frame is valid (atoms not colinear).
    """
    if len(frame_atom_indices) != 3:
        return False

    positions = atom_positions[frame_atom_indices]
    return not check_colinear(positions)
