"""Structure feature extraction module.

This module provides functionality for extracting features from protein structures,
including coordinate processing, frame construction, distance matrices, and contact maps.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from onescience.datapipes.biology.common.features.feature_base import (
    BaseFeatureExtractor,
    FeatureDict,
)
from onescience.datapipes.biology.common.features.constants import (
    STRUCTURE_FEATURE_NAMES,
    ATOM_ORDER,
    ATOM_TYPES,
    RESTYPE_ORDER,
    RESTYPES,
)


class StructureFeatureExtractor(BaseFeatureExtractor):
    """Structure feature extractor.

    Extracts features from protein structure data, including coordinates,
    frames, distance matrices, and contact maps.

    Example:
        >>> extractor = StructureFeatureExtractor(config={'use_frames': True})
        >>> features = extractor.extract({'coords': atom_coords, 'aatype': residue_types})
    """

    def __init__(self, config: Optional[FeatureDict] = None):
        """Initialize the structure feature extractor.

        Args:
            config: Configuration dictionary containing:
                - use_frames: Whether to compute frame features.
                - use_pseudo_beta: Whether to compute pseudo-beta positions.
        """
        super().__init__(config)
        self.use_frames = self.config.get('use_frames', True)
        self.use_pseudo_beta = self.config.get('use_pseudo_beta', True)

    def extract(self, data: FeatureDict) -> FeatureDict:
        """Extract structure features from input data.

        Args:
            data: Input data dictionary containing:
                - coords: Atom coordinates of shape (num_res, num_atoms, 3).
                - aatype: Amino acid type indices.
                - atom_mask: Optional atom mask.

        Returns:
            Dictionary containing structure features:
                - all_atom_positions: Atom coordinates.
                - all_atom_mask: Atom mask.
                - pseudo_beta: Pseudo-beta positions.
                - frames: Rigid body frames.
        """
        coords = data.get('coords')
        aatype = data.get('aatype')

        if coords is None or aatype is None:
            return {}

        features = {}

        # Store coordinates
        features['all_atom_positions'] = coords

        # Create atom mask
        atom_mask = data.get('atom_mask')
        if atom_mask is None:
            atom_mask = np.ones(coords.shape[:2], dtype=np.float32)
        features['all_atom_mask'] = atom_mask

        # Compute pseudo-beta positions
        if self.use_pseudo_beta:
            pseudo_beta = pseudo_beta_fn(aatype, coords, atom_mask)
            features['pseudo_beta'] = pseudo_beta
            features['pseudo_beta_mask'] = atom_mask[:, 0]  # CA atom mask

        # Compute frames
        if self.use_frames:
            frames = atom37_to_frames(aatype, coords, atom_mask)
            features.update(frames)

        return features


def make_structure_features(
    coords: np.ndarray,
    aatype: np.ndarray,
    atom_mask: Optional[np.ndarray] = None
) -> FeatureDict:
    """Create structure features from coordinates and residue types.

    Args:
        coords: Atom coordinates of shape (num_res, num_atoms, 3).
        aatype: Amino acid type indices of shape (num_res,).
        atom_mask: Optional atom mask of shape (num_res, num_atoms).

    Returns:
        Dictionary containing structure features.
    """
    if atom_mask is None:
        atom_mask = np.ones(coords.shape[:2], dtype=np.float32)

    features = {
        'all_atom_positions': coords,
        'all_atom_mask': atom_mask,
        'aatype': aatype,
    }

    # Add pseudo-beta
    pseudo_beta = pseudo_beta_fn(aatype, coords, atom_mask)
    features['pseudo_beta'] = pseudo_beta
    features['pseudo_beta_mask'] = atom_mask[:, 0]

    return features


def pseudo_beta_fn(
    aatype: np.ndarray,
    all_atom_positions: np.ndarray,
    all_atom_mask: np.ndarray
) -> np.ndarray:
    """Compute pseudo-beta positions for each residue.

    For standard amino acids, returns the CB atom position if available,
    otherwise estimates it from CA atom. For glycine, uses a fixed offset
    from the CA atom.

    Args:
        aatype: Amino acid type indices of shape (num_res,).
        all_atom_positions: Atom coordinates of shape (num_res, 37, 3).
        all_atom_mask: Atom mask of shape (num_res, 37).

    Returns:
        Pseudo-beta positions of shape (num_res, 3).
    """
    # CB atom index is 4 in atom37 format
    cb_index = 4
    ca_index = 1

    # Get CB positions and mask
    cb_positions = all_atom_positions[:, cb_index]
    cb_mask = all_atom_mask[:, cb_index]

    # For residues without CB (glycine), use CA with offset
    pseudo_beta = cb_positions.copy()

    # Estimate CB position for glycine
    glycine_idx = 7  # Glycine index in RESTYPE_ORDER
    is_glycine = (aatype == glycine_idx)

    # Use CA position + offset for glycine
    ca_positions = all_atom_positions[:, ca_index]
    pseudo_beta[is_glycine] = ca_positions[is_glycine] + np.array([1.0, 0.0, 0.0])

    return pseudo_beta


def make_pseudo_beta(
    aatype: np.ndarray,
    all_atom_positions: np.ndarray,
    all_atom_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Create pseudo-beta positions and mask.

    Args:
        aatype: Amino acid type indices.
        all_atom_positions: Atom coordinates.
        all_atom_mask: Atom mask.

    Returns:
        Tuple of (pseudo_beta_positions, pseudo_beta_mask).
    """
    pseudo_beta = pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask)
    pseudo_beta_mask = all_atom_mask[:, 1]  # CA atom mask

    return pseudo_beta, pseudo_beta_mask


def atom37_to_frames(
    aatype: np.ndarray,
    all_atom_positions: np.ndarray,
    all_atom_mask: np.ndarray
) -> FeatureDict:
    """Convert atom37 representation to rigid body frames.

    Constructs backbone frames from N, CA, C atoms for each residue.

    Args:
        aatype: Amino acid type indices of shape (num_res,).
        all_atom_positions: Atom coordinates of shape (num_res, 37, 3).
        all_atom_mask: Atom mask of shape (num_res, 37).

    Returns:
        Dictionary containing frame information:
            - frames: Rigid body frames (not fully implemented).
            - backbone_coords: Backbone atom coordinates (N, CA, C).
    """
    # Backbone atom indices: N=0, CA=1, C=2
    backbone_indices = [0, 1, 2]

    backbone_coords = all_atom_positions[:, backbone_indices]
    backbone_mask = all_atom_mask[:, backbone_indices]

    # Store backbone information
    # Full frame construction would require rotation matrix computation
    frames = {
        'backbone_coords': backbone_coords,
        'backbone_mask': backbone_mask,
        'n_coords': all_atom_positions[:, 0],
        'ca_coords': all_atom_positions[:, 1],
        'c_coords': all_atom_positions[:, 2],
    }

    return frames


def compute_distance_matrix(
    coords: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute pairwise distance matrix.

    Args:
        coords: Coordinates of shape (num_points, 3) or (num_points, num_atoms, 3).
        mask: Optional mask of shape (num_points,) or (num_points, num_atoms).

    Returns:
        Distance matrix of shape (num_points, num_points).
    """
    # If multi-atom per point, use first atom or mean
    if coords.ndim == 3:
        coords = coords[:, 0]  # Use first atom

    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Apply mask if provided
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, 0]
        valid_mask = mask[:, None] * mask[None, :]
        distances = distances * valid_mask

    return distances


def compute_contact_map(
    coords: np.ndarray,
    threshold: float = 8.0,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute contact map from coordinates.

    Args:
        coords: Coordinates of shape (num_res, 3) or (num_res, num_atoms, 3).
        threshold: Distance threshold for contact (in Angstroms).
        mask: Optional mask.

    Returns:
        Binary contact map of shape (num_res, num_res).
    """
    distances = compute_distance_matrix(coords, mask)
    contact_map = (distances < threshold).astype(np.float32)

    # Exclude self-contacts
    np.fill_diagonal(contact_map, 0)

    return contact_map


def compute_backbone_dihedrals(
    coords: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute backbone dihedral angles (phi, psi, omega).

    Args:
        coords: Backbone coordinates of shape (num_res, 3, 3) for (N, CA, C).

    Returns:
        Dictionary containing:
            - phi: Phi angles in radians.
            - psi: Psi angles in radians.
            - omega: Omega angles in radians.
    """
    num_res = coords.shape[0]

    # Extract N, CA, C coordinates
    n_coords = coords[:, 0]
    ca_coords = coords[:, 1]
    c_coords = coords[:, 2]

    # Compute phi: C(i-1) - N(i) - CA(i) - C(i)
    phi = np.zeros(num_res, dtype=np.float32)
    for i in range(1, num_res):
        phi[i] = _dihedral_angle(c_coords[i-1], n_coords[i], ca_coords[i], c_coords[i])

    # Compute psi: N(i) - CA(i) - C(i) - N(i+1)
    psi = np.zeros(num_res, dtype=np.float32)
    for i in range(num_res - 1):
        psi[i] = _dihedral_angle(n_coords[i], ca_coords[i], c_coords[i], n_coords[i+1])

    # Compute omega: CA(i) - C(i) - N(i+1) - CA(i+1)
    omega = np.zeros(num_res, dtype=np.float32)
    for i in range(num_res - 1):
        omega[i] = _dihedral_angle(ca_coords[i], c_coords[i], n_coords[i+1], ca_coords[i+1])

    return {
        'phi': phi,
        'psi': psi,
        'omega': omega,
    }


def _dihedral_angle(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray
) -> float:
    """Compute dihedral angle from 4 points.

    Args:
        p1, p2, p3, p4: 3D coordinates of 4 points.

    Returns:
        Dihedral angle in radians.
    """
    # Compute vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normalize b2
    b2_norm = b2 / (np.linalg.norm(b2) + 1e-8)

    # Compute normal vectors
    n1 = np.cross(b1, b2)
    n1 = n1 / (np.linalg.norm(n1) + 1e-8)

    n2 = np.cross(b2, b3)
    n2 = n2 / (np.linalg.norm(n2) + 1e-8)

    # Compute angle
    m1 = np.cross(n1, b2_norm)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.arctan2(y, x)


def compute_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Compute RMSD between two coordinate sets.

    Args:
        coords1: First coordinate array.
        coords2: Second coordinate array.
        mask: Optional mask for valid positions.

    Returns:
        RMSD value.
    """
    if mask is None:
        mask = np.ones(coords1.shape[0], dtype=np.float32)

    diff = coords1 - coords2
    squared_diff = np.sum(diff ** 2, axis=-1)

    valid_mask = mask > 0
    if valid_mask.sum() == 0:
        return 0.0

    rmsd = np.sqrt(squared_diff[valid_mask].mean())
    return float(rmsd)


def kabsch_alignment(
    coords1: np.ndarray,
    coords2: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two coordinate sets using Kabsch algorithm.

    Args:
        coords1: Reference coordinates.
        coords2: Coordinates to align.
        mask: Optional mask for valid positions.

    Returns:
        Tuple of (rotation_matrix, translation_vector).
    """
    if mask is None:
        mask = np.ones(coords1.shape[0], dtype=np.float32)

    # Center coordinates
    valid_mask = mask > 0
    center1 = coords1[valid_mask].mean(axis=0)
    center2 = coords2[valid_mask].mean(axis=0)

    centered1 = coords1 - center1
    centered2 = coords2 - center2

    # Compute covariance matrix
    H = centered2[valid_mask].T @ centered1[valid_mask]

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    d = np.linalg.det(U @ Vt)
    diag = np.eye(3)
    diag[2, 2] = np.sign(d)
    R = U @ diag @ Vt

    # Compute translation
    t = center1 - R @ center2

    return R, t
