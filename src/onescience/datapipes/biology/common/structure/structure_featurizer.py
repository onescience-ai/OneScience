"""Module for unified structure feature extraction.

Extracts features from structure objects.
"""

from typing import Dict, Optional
import numpy as np
from onescience.datapipes.biology.common.structure.structure_parser import Structure


class StructureFeaturizer:
    """Unified structure feature extractor.

    Extracted features include:
    - Atomic coordinates
    - Atom masks
    - Distance matrices
    - Angle features
    """
    
    def __init__(self, atom_types: Optional[list] = None):
        """Initialize the feature extractor.

        Args:
            atom_types: List of atom types to extract (e.g., ['CA', 'C', 'N', 'O']).
                If None, extracts all atoms.
        """
        self.atom_types = atom_types or ['CA', 'C', 'N', 'O', 'CB']
    
    def featurize(self, structure: Structure, chain_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Extract structure features.

        Args:
            structure: Structure object.
            chain_id: If specified, only extract features for this chain.

        Returns:
            Dictionary of features.
        """
        features = {}
        
        # Filter atoms (if chain specified)
        if chain_id:
            atoms = [atom for atom in structure.atoms if atom.chain_id == chain_id]
        else:
            atoms = structure.atoms
        
        # Extract coordinates and masks for each atom type
        all_atom_positions = []
        all_atom_mask = []
        
        for atom_type in self.atom_types:
            positions = []
            mask = []
            
            for atom in atoms:
                if atom.name == atom_type:
                    positions.append([atom.x, atom.y, atom.z])
                    mask.append(1.0)
                else:
                    positions.append([0.0, 0.0, 0.0])
                    mask.append(0.0)
            
            if positions:
                all_atom_positions.append(positions)
                all_atom_mask.append(mask)
        
        if all_atom_positions:
            # Shape: (num_atom_types, num_residues, 3)
            features['all_atom_positions'] = np.array(all_atom_positions, dtype=np.float32)
            features['all_atom_mask'] = np.array(all_atom_mask, dtype=np.float32)
        
        # Extract CA atom coordinates (for distance matrix calculation)
        ca_positions = structure.get_atom_positions('CA')
        if chain_id:
            ca_atoms = [atom for atom in structure.atoms 
                       if atom.chain_id == chain_id and atom.name == 'CA']
            ca_positions = np.array([[atom.x, atom.y, atom.z] for atom in ca_atoms], dtype=np.float32)
        
        if len(ca_positions) > 0:
            # Distance matrix
            features['ca_distance_matrix'] = self._compute_distance_matrix(ca_positions)

            # CA atom mask
            features['ca_mask'] = np.ones(len(ca_positions), dtype=np.float32)
        else:
            features['ca_distance_matrix'] = np.array([], dtype=np.float32).reshape(0, 0)
            features['ca_mask'] = np.array([], dtype=np.float32)
        
        return features
    
    def _compute_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """Compute distance matrix.

        Args:
            positions: Array of positions with shape (num_atoms, 3).

        Returns:
            Distance matrix with shape (num_atoms, num_atoms).
        """
        if len(positions) == 0:
            return np.array([], dtype=np.float32).reshape(0, 0)
        
        # Compute pairwise distances
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))
        return distances.astype(np.float32)





