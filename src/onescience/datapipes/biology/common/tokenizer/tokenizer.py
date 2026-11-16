"""
通用AtomArray Tokenizer

参考AlphaFold3 SI Chapter 2.6实现
将AtomArray对象转换为TokenArray，支持标准残基和配体的token化
"""

from typing import List, Optional, Dict, Any
import numpy as np
from biotite.structure import AtomArray
import biotite.structure as struc


class Token:
    """
    Token对象，用于存储与token相关的信息

    在AlphaFold3中：
    - 标准残基（蛋白质、DNA、RNA）：每个残基一个token
    - 配体/非标准残基：每个原子一个token

    Example:
        >>> token = Token(1)
        >>> token.value
        1
        >>> token.atom_indices = [1, 2, 3]
        >>> token.centre_atom_index = 2
    """

    def __init__(self, value: int, **kwargs):
        """
        初始化Token

        Args:
            value: token值（通常是残基类型索引）
            **kwargs: 其他属性
        """
        self.value = value
        self._annot: Dict[str, Any] = {}
        for name, annotation in kwargs.items():
            self._annot[name] = annotation

    def __getattr__(self, attr: str) -> Any:
        """获取属性，先从_annot中查找"""
        if attr in super().__getattribute__("_annot"):
            return self._annot[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def __repr__(self) -> str:
        """字符串表示"""
        annot_lst = [f"{k}={v}" for k, v in self._annot.items()]
        return f'Token({self.value}, {",".join(annot_lst)})'

    def __setattr__(self, attr: str, value: Any) -> None:
        """设置属性，_annot和value特殊处理"""
        if attr == "_annot":
            super().__setattr__(attr, value)
        elif attr == "value":
            super().__setattr__(attr, value)
        else:
            self._annot[attr] = value

    def get_annotation(self, key: str) -> Any:
        """获取标注"""
        return self._annot.get(key)

    def set_annotation(self, key: str, value: Any) -> None:
        """设置标注"""
        self._annot[key] = value


class TokenArray:
    """
    Token数组，用于批量操作一组Token对象
    """

    def __init__(self, tokens: List[Token]):
        """
        初始化TokenArray

        Args:
            tokens: Token对象列表
        """
        self.tokens = tokens

    def __repr__(self) -> str:
        """字符串表示"""
        repr_str = "TokenArray(\n"
        for token in self.tokens:
            repr_str += f"\t{token}\n"
        repr_str += ")"
        return repr_str

    def __len__(self) -> int:
        """返回token数量"""
        return len(self.tokens)

    def __iter__(self):
        """迭代器"""
        for token in self.tokens:
            yield token

    def __getitem__(self, index):
        """索引访问"""
        if isinstance(index, int):
            return self.tokens[index]
        else:
            return TokenArray([self.tokens[i] for i in index])

    def get_annotation(self, category: str) -> List[Any]:
        """
        获取所有token的某个标注

        Args:
            category: 标注类别

        Returns:
            List: 所有token的该标注值列表
        """
        return [token._annot.get(category) for token in self.tokens]

    def set_annotation(self, category: str, values: List[Any]) -> None:
        """
        为所有token设置某个标注

        Args:
            category: 标注类别
            values: 标注值列表，长度必须等于token数量
        """
        assert len(values) == len(
            self.tokens
        ), f"Length of values ({len(values)}) must match the number of tokens ({len(self.tokens)})"
        for token, value in zip(self.tokens, values):
            token._annot[category] = value

    def get_values(self) -> List[int]:
        """获取所有token的值"""
        return [token.value for token in self.tokens]

    def get_atom_indices(self) -> List[List[int]]:
        """获取所有token包含的原子索引"""
        return [token.atom_indices for token in self.tokens]

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            Dict: 包含所有token信息的字典
        """
        return {
            "num_tokens": len(self.tokens),
            "values": self.get_values(),
            "atom_indices": self.get_atom_indices(),
            "annotations": {
                key: self.get_annotation(key)
                for key in self.tokens[0]._annot.keys()
            } if self.tokens else {}
        }


class AtomArrayTokenizer:
    """
    AtomArray Tokenizer

    将AtomArray对象token化为TokenArray

    Ref: AlphaFold3 SI Chapter 2.6
    - 标准残基（蛋白质、DNA、RNA）：每个残基一个token
    - 配体和非标准残基：每个重原子一个token
    """

    def __init__(
        self,
        atom_array: AtomArray,
        std_residues: Optional[Dict[str, int]] = None,
        elems: Optional[Dict[str, int]] = None,
    ):
        """
        初始化Tokenizer

        Args:
            atom_array: Biotite AtomArray对象
            std_residues: 标准残基到token值的映射（可选）
            elems: 元素到token值的映射（可选）
        """
        self.atom_array = atom_array

        # 默认标准残基定义（AlphaFold3风格）
        if std_residues is None:
            from onescience.datapipes.biology.common.features.constants import (
                STD_RESIDUES,
            )
            self.std_residues = STD_RESIDUES
        else:
            self.std_residues = std_residues

        # 默认元素定义
        if elems is None:
            from onescience.datapipes.biology.common.features.constants import ELEMS
            self.elems = ELEMS
        else:
            self.elems = elems

    def tokenize(self) -> List[Token]:
        """
        Token化AtomArray

        Returns:
            List[Token]: Token对象列表
        """
        tokens = []
        total_atom_num = 0

        for res in struc.residue_iter(self.atom_array):
            atom_num = len(res)
            first_atom = res[0]
            res_name = first_atom.res_name
            mol_type = getattr(first_atom, "mol_type", "protein")

            # 获取标准残基的token值
            res_token = self.std_residues.get(res_name, None)

            if res_token is not None and mol_type != "ligand":
                # 标准残基：每个残基一个token
                token = Token(res_token)
                atom_indices = list(range(total_atom_num, total_atom_num + atom_num))
                atom_names = [self.atom_array[i].atom_name for i in atom_indices]

                token.atom_indices = atom_indices
                token.atom_names = atom_names
                tokens.append(token)
                total_atom_num += atom_num
            else:
                # 配体和非标准残基：每个原子一个token
                for atom in res:
                    atom_elem = atom.element
                    atom_token = self.elems.get(atom_elem, None)

                    if atom_token is None:
                        # 未知元素，使用默认值
                        atom_token = max(self.elems.values()) + 1 if self.elems else 128

                    token = Token(atom_token)
                    token.atom_indices = [total_atom_num]
                    token.atom_names = [atom.atom_name]
                    tokens.append(token)
                    total_atom_num += 1

        assert total_atom_num == len(self.atom_array), \
            f"Tokenization mismatch: {total_atom_num} vs {len(self.atom_array)}"

        return tokens

    def _set_token_annotations(self, token_array: TokenArray) -> TokenArray:
        """
        设置token标注

        Args:
            token_array: TokenArray对象

        Returns:
            TokenArray: 带有标注的TokenArray
        """
        # 获取中心原子索引（centre_atom_mask == 1的位置）
        if hasattr(self.atom_array, "centre_atom_mask"):
            centre_atom_indices = np.where(self.atom_array.centre_atom_mask == 1)[0]
        else:
            # 如果没有centre_atom_mask，使用每个残基的第一个原子
            centre_atom_indices = self._get_default_centre_atom_indices()

        token_array.set_annotation("centre_atom_index", centre_atom_indices.tolist())
        assert len(token_array) == len(centre_atom_indices), \
            f"Token count mismatch: {len(token_array)} vs {len(centre_atom_indices)}"

        return token_array

    def _get_default_centre_atom_indices(self) -> np.ndarray:
        """
        获取默认的中心原子索引

        当centre_atom_mask不存在时使用：
        - 蛋白质：CA原子
        - DNA/RNA：C1'原子
        - 其他：第一个原子

        Returns:
            np.ndarray: 中心原子索引数组
        """
        centre_indices = []

        for res in struc.residue_iter(self.atom_array):
            res_atoms = res
            atom_names = [atom.atom_name for atom in res_atoms]
            global_start_idx = next(iter(res_atoms)).array_index(0)

            # 查找中心原子
            if "CA" in atom_names:
                idx = atom_names.index("CA")
            elif "C1'" in atom_names:
                idx = atom_names.index("C1'")
            else:
                idx = 0

            centre_indices.append(global_start_idx + idx)

        return np.array(centre_indices, dtype=np.int64)

    def get_token_array(self) -> TokenArray:
        """
        获取带有标注的TokenArray

        Returns:
            TokenArray: 包含atom_indices, centre_atom_index等标注的TokenArray

        Example:
            TokenArray(
                Token(1, atom_indices=[0,1,2,...], centre_atom_index=2, atom_names=['N','CA','C',...])
                Token(15, atom_indices=[11,12,...], centre_atom_index=13, atom_names=['N','CA',...])
            )
        """
        tokens = self.tokenize()
        token_array = TokenArray(tokens=tokens)
        token_array = self._set_token_annotations(token_array=token_array)
        return token_array


def create_token_array(
    atom_array: AtomArray,
    use_centre_atom_mask: bool = True,
) -> TokenArray:
    """
    便捷函数：从AtomArray创建TokenArray

    Args:
        atom_array: Biotite AtomArray对象
        use_centre_atom_mask: 是否使用centre_atom_mask（如果存在）

    Returns:
        TokenArray: TokenArray对象
    """
    tokenizer = AtomArrayTokenizer(atom_array)
    return tokenizer.get_token_array()


def token_array_to_atom_indices(token_array: TokenArray) -> List[int]:
    """
    获取TokenArray中所有token的中心原子索引

    Args:
        token_array: TokenArray对象

    Returns:
        List[int]: 中心原子索引列表
    """
    return token_array.get_annotation("centre_atom_index")


def get_token_atom_mapping(token_array: TokenArray, num_atoms: int) -> np.ndarray:
    """
    获取原子到token的映射数组

    Args:
        token_array: TokenArray对象
        num_atoms: 总原子数

    Returns:
        np.ndarray: 原子到token的映射，shape: [N_atom]，值为token索引
    """
    atom_to_token = np.full(num_atoms, -1, dtype=np.int64)

    for token_idx, token in enumerate(token_array):
        for atom_idx in token.atom_indices:
            if 0 <= atom_idx < num_atoms:
                atom_to_token[atom_idx] = token_idx

    return atom_to_token
