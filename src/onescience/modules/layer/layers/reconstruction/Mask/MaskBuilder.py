import torch
import torch.nn as nn

from onescience.registry import Mask

@Mask.registry_module()
class MaskBuilder:
    """
    MaskBuilder
    ======================================================
    功能:
        - 根据输入序列 / MSA /原子信息生成掩码
        - 支持 Residue Mask、Atom Mask、Pair Mask
    """

    @staticmethod
    def build_residue_mask(seq, pad_idx=0):
        """
        构建残基有效性掩码
        seq: [B, L] 序列编码，pad_idx表示填充
        return: [B, L] 1=有效残基, 0=padding
        """
        return (seq != pad_idx).float()

    @staticmethod
    def build_atom_mask(residue_mask, num_atoms=3):
        """
        根据残基掩码构建原子掩码
        residue_mask: [B, L] 残基掩码
        return: [B, L, num_atoms]
        """
        return residue_mask.unsqueeze(-1).repeat(1, 1, num_atoms)

    @staticmethod
    def build_pair_mask(residue_mask):
        """
        构建残基对掩码
        residue_mask: [B, L]
        return: [B, L, L]
        """
        return residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)


if __name__ == "__main__":
    B, L = 2, 5
    seq = torch.tensor([
        [1,2,3,0,0],
        [4,5,0,0,0]
    ])

    # 残基掩码
    residue_mask = MaskBuilder.build_residue_mask(seq)
    print("Residue mask:", residue_mask)

    # 原子掩码 (假设3个主链原子 N,CA,C)
    atom_mask = MaskBuilder.build_atom_mask(residue_mask, num_atoms=3)
    print("Atom mask:", atom_mask)

    # 残基对掩码
    pair_mask = MaskBuilder.build_pair_mask(residue_mask)
    print("Pair mask:", pair_mask)
