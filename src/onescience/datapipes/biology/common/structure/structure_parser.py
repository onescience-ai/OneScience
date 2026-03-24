"""
统一的结构文件解析器

支持mmCIF和PDB格式
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

from onescience.datapipes.biology.common.utils.file_utils import (
    open_file,
    detect_file_format,
)


@dataclass
class Atom:
    """原子信息"""
    name: str
    element: str
    residue_name: str
    residue_number: int
    chain_id: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0


@dataclass
class Structure:
    """
    统一的结构数据格式
    
    Attributes
    ----------
    atoms : List[Atom]
        原子列表
    chains : List[str]
        链ID列表
    sequence : Dict[str, str]
        每个链的序列（链ID -> 序列）
    """
    atoms: List[Atom]
    chains: List[str]
    sequence: Dict[str, str]
    
    def get_atom_positions(self, atom_name: Optional[str] = None) -> np.ndarray:
        """
        获取原子坐标
        
        Parameters
        ----------
        atom_name : Optional[str]
            如果指定，只返回该原子的坐标（如'CA'）
            
        Returns
        -------
        np.ndarray
            Shape: (num_atoms, 3)
        """
        if atom_name:
            positions = [
                [atom.x, atom.y, atom.z]
                for atom in self.atoms
                if atom.name == atom_name
            ]
        else:
            positions = [
                [atom.x, atom.y, atom.z]
                for atom in self.atoms
            ]
        return np.array(positions, dtype=np.float32)
    
    def get_atom_mask(self, atom_name: Optional[str] = None) -> np.ndarray:
        """
        获取原子掩码
        
        Parameters
        ----------
        atom_name : Optional[str]
            如果指定，只返回该原子的掩码
            
        Returns
        -------
        np.ndarray
            Shape: (num_atoms,)
        """
        if atom_name:
            mask = [
                1.0 if atom.name == atom_name else 0.0
                for atom in self.atoms
            ]
        else:
            mask = [1.0] * len(self.atoms)
        return np.array(mask, dtype=np.float32)


class StructureParser:
    """
    统一的结构文件解析器
    
    支持格式：
    - mmCIF (.cif)
    - PDB (.pdb)
    """
    
    @staticmethod
    def parse_mmcif(cif_string: str) -> Structure:
        """
        解析mmCIF格式
        
        Parameters
        ----------
        cif_string : str
            mmCIF格式的字符串
            
        Returns
        -------
        Structure
            解析后的结构对象
        """
        atoms = []
        chains = []
        sequence = {}
        
        current_chain = None
        current_residue = None
        current_seq = ""
        
        in_atom_site = False
        
        for line in cif_string.splitlines():
            line = line.strip()
            
            if line.startswith("_atom_site."):
                in_atom_site = True
                continue
            elif line.startswith("#") or not line:
                continue
            elif in_atom_site and not line.startswith("_"):
                # 解析原子行
                parts = line.split()
                if len(parts) >= 11:
                    # 标准mmCIF格式：group_PDB, id, type_symbol, label_atom_id, 
                    # label_alt_id, label_comp_id, label_asym_id, label_entity_id,
                    # label_seq_id, Cartn_x, Cartn_y, Cartn_z
                    try:
                        atom_name = parts[3]
                        residue_name = parts[5]
                        chain_id = parts[6]
                        residue_number = int(parts[8])
                        x = float(parts[9])
                        y = float(parts[10])
                        z = float(parts[11]) if len(parts) > 11 else 0.0
                        
                        # 提取元素（从原子名推断）
                        element = atom_name[0] if atom_name else "C"
                        
                        atom = Atom(
                            name=atom_name,
                            element=element,
                            residue_name=residue_name,
                            residue_number=residue_number,
                            chain_id=chain_id,
                            x=x,
                            y=y,
                            z=z,
                        )
                        atoms.append(atom)
                        
                        # 更新链信息
                        if chain_id not in chains:
                            chains.append(chain_id)
                            sequence[chain_id] = ""
                        
                        # 更新序列（只记录CA原子）
                        if atom_name == "CA" and (current_chain != chain_id or current_residue != residue_number):
                            current_chain = chain_id
                            current_residue = residue_number
                            # 将三字母代码转换为单字母代码（简化版）
                            aa_code = StructureParser._three_to_one(residue_name)
                            sequence[chain_id] += aa_code
                    except (ValueError, IndexError):
                        continue
        
        return Structure(
            atoms=atoms,
            chains=chains,
            sequence=sequence,
        )
    
    @staticmethod
    def parse_pdb(pdb_string: str) -> Structure:
        """
        解析PDB格式
        
        Parameters
        ----------
        pdb_string : str
            PDB格式的字符串
            
        Returns
        -------
        Structure
            解析后的结构对象
        """
        atoms = []
        chains = []
        sequence = {}
        
        current_chain = None
        current_residue = None
        
        for line in pdb_string.splitlines():
            line = line.strip()
            
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # 解析ATOM行
                # 格式：ATOM  serial  name  altLoc  resName  chainID  resSeq  iCode  x  y  z  occupancy  tempFactor
                try:
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip() or "A"
                    residue_number = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    occupancy = 1.0
                    if len(line) > 54:
                        try:
                            occupancy = float(line[54:60].strip())
                        except ValueError:
                            pass
                    
                    b_factor = 0.0
                    if len(line) > 60:
                        try:
                            b_factor = float(line[60:66].strip())
                        except ValueError:
                            pass
                    
                    # 提取元素（从原子名推断）
                    element = atom_name[0] if atom_name else "C"
                    
                    atom = Atom(
                        name=atom_name,
                        element=element,
                        residue_name=residue_name,
                        residue_number=residue_number,
                        chain_id=chain_id,
                        x=x,
                        y=y,
                        z=z,
                        occupancy=occupancy,
                        b_factor=b_factor,
                    )
                    atoms.append(atom)
                    
                    # 更新链信息
                    if chain_id not in chains:
                        chains.append(chain_id)
                        sequence[chain_id] = ""
                    
                    # 更新序列（只记录CA原子）
                    if atom_name == "CA" and (current_chain != chain_id or current_residue != residue_number):
                        current_chain = chain_id
                        current_residue = residue_number
                        # 将三字母代码转换为单字母代码
                        aa_code = StructureParser._three_to_one(residue_name)
                        sequence[chain_id] += aa_code
                except (ValueError, IndexError):
                    continue
        
        return Structure(
            atoms=atoms,
            chains=chains,
            sequence=sequence,
        )
    
    @staticmethod
    def _three_to_one(residue_name: str) -> str:
        """
        将三字母氨基酸代码转换为单字母代码
        
        Parameters
        ----------
        residue_name : str
            三字母代码（如'ALA'）
            
        Returns
        -------
        str
            单字母代码（如'A'）
        """
        three_to_one_map = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
            'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
            'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
            'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        }
        return three_to_one_map.get(residue_name.upper(), 'X')
    
    @staticmethod
    def parse_file(path: Path, format: Optional[str] = None) -> Structure:
        """
        从文件解析结构（支持压缩文件）
        
        Parameters
        ----------
        path : Path
            结构文件路径（支持.gz, .bz2, .xz压缩文件）
        format : Optional[str]
            格式（"mmcif" 或 "pdb"），如果为None则自动检测
            
        Returns
        -------
        Structure
            解析后的结构对象
        """
        with open_file(path, 'r') as f:
            content = f.read()
        
        # 自动检测格式
        if format is None:
            detected_format = detect_file_format(path)
            if detected_format in ['mmcif', 'cif']:
                format = 'mmcif'
            elif detected_format == 'pdb':
                format = 'pdb'
            elif 'data_' in content[:100] and '_atom_site' in content:
                format = 'mmcif'
            elif content.startswith('HEADER') or 'ATOM' in content[:100]:
                format = 'pdb'
            else:
                raise ValueError(
                    f"Could not detect structure format for file: {path}. "
                    f"Supported formats: mmcif, pdb"
                )
        
        if format == 'mmcif':
            return StructureParser.parse_mmcif(content)
        elif format == 'pdb':
            return StructureParser.parse_pdb(content)
        else:
            raise ValueError(
                f"Unsupported structure format: {format}. "
                f"Supported formats: mmcif, pdb"
            )

