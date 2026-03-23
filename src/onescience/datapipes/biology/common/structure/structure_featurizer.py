"""
统一的结构特征提取器

从结构对象中提取特征
"""

from typing import Dict, Optional
import numpy as np
from onescience.datapipes.biology.common.structure.structure_parser import Structure


class StructureFeaturizer:
    """
    统一的结构特征提取器
    
    提取的特征包括：
    - 原子坐标
    - 原子掩码
    - 距离矩阵
    - 角度特征
    """
    
    def __init__(self, atom_types: Optional[list] = None):
        """
        Parameters
        ----------
        atom_types : Optional[list]
            要提取的原子类型列表（如['CA', 'C', 'N', 'O']），如果为None则提取所有原子
        """
        self.atom_types = atom_types or ['CA', 'C', 'N', 'O', 'CB']
    
    def featurize(self, structure: Structure, chain_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        提取结构特征
        
        Parameters
        ----------
        structure : Structure
            结构对象
        chain_id : Optional[str]
            如果指定，只提取该链的特征
            
        Returns
        -------
        Dict[str, np.ndarray]
            特征字典
        """
        features = {}
        
        # 过滤原子（如果指定了链）
        if chain_id:
            atoms = [atom for atom in structure.atoms if atom.chain_id == chain_id]
        else:
            atoms = structure.atoms
        
        # 提取每种原子类型的坐标和掩码
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
        
        # 提取CA原子坐标（用于距离矩阵计算）
        ca_positions = structure.get_atom_positions('CA')
        if chain_id:
            ca_atoms = [atom for atom in structure.atoms 
                       if atom.chain_id == chain_id and atom.name == 'CA']
            ca_positions = np.array([[atom.x, atom.y, atom.z] for atom in ca_atoms], dtype=np.float32)
        
        if len(ca_positions) > 0:
            # 距离矩阵
            features['ca_distance_matrix'] = self._compute_distance_matrix(ca_positions)
            
            # CA原子掩码
            features['ca_mask'] = np.ones(len(ca_positions), dtype=np.float32)
        else:
            features['ca_distance_matrix'] = np.array([], dtype=np.float32).reshape(0, 0)
            features['ca_mask'] = np.array([], dtype=np.float32)
        
        return features
    
    def _compute_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        计算距离矩阵
        
        Parameters
        ----------
        positions : np.ndarray
            Shape: (num_atoms, 3)
            
        Returns
        -------
        np.ndarray
            Shape: (num_atoms, num_atoms)
        """
        if len(positions) == 0:
            return np.array([], dtype=np.float32).reshape(0, 0)
        
        # 计算所有点对之间的距离
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))
        return distances.astype(np.float32)





