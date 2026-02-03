import torch


from onescience.registry import Mask

@Mask.registry_module()
class Atom14Mapping:
    """
    Atom14Mapping
    ======================================================
    功能:
        - 将残基原子映射到固定 14 个槽位
        - 便于构建原子特征张量，适配 Transformer / Structure Module
        - 支持主链、侧链原子映射
    """

    # 常用残基14原子槽位索引表（示例）
    ATOM14_NAMES = [
        "N", "CA", "C", "O", "CB",
        "CG", "CG1", "CG2", "CD", "CD1",
        "CD2", "CE", "CE1", "CE2"
    ]

    def __init__(self, residue_to_atom_map):
        """
        residue_to_atom_map: dict
            key=residue_name (str), value=dict(atom_name->index in 14 slots)
        """
        self.residue_to_atom_map = residue_to_atom_map

    def map_atoms(self, residue_name, atom_coords):
        """
        residue_name: str 残基类型
        atom_coords: dict {atom_name: [3]} 原子坐标
        return:
            mapped_coords: [14,3] 填充缺失原子为0
            atom_mask: [14] 1=原子存在, 0=缺失
        """
        mapped_coords = torch.zeros(14, 3)
        atom_mask = torch.zeros(14)

        if residue_name not in self.residue_to_atom_map:
            return mapped_coords, atom_mask

        atom_index_map = self.residue_to_atom_map[residue_name]
        for atom_name, slot in atom_index_map.items():
            if atom_name in atom_coords:
                mapped_coords[slot] = torch.tensor(atom_coords[atom_name])
                atom_mask[slot] = 1.0

        return mapped_coords, atom_mask


if __name__ == "__main__":
    # 简化残基映射示例
    residue_to_atom_map = {
        "ALA": {"N":0, "CA":1, "C":2, "O":3, "CB":4},
        "VAL": {"N":0, "CA":1, "C":2, "O":3, "CB":4, "CG1":5, "CG2":6}
    }

    atom_coords = {
        "N": [0.0,0.1,0.2],
        "CA": [1.0,0.0,0.0],
        "C": [1.5,0.5,0.0],
        "O": [1.7,0.8,0.1],
        "CB": [0.9,1.0,0.0]
    }

    mapping = Atom14Mapping(residue_to_atom_map)
    coords, mask = mapping.map_atoms("ALA", atom_coords)
    print("Mapped coords:", coords)
    print("Atom mask:", mask)
