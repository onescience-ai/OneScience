"""General AtomArray Tokenizer.

Implements tokenization following AlphaFold3 SI Chapter 2.6.
Converts AtomArray objects to TokenArray, supporting tokenization of
standard residues and ligands.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from biotite.structure import AtomArray
import biotite.structure as struc


class Token:
    """Class for storing token-related information.

    In AlphaFold3:
    - Standard residues (protein, DNA, RNA): one token per residue
    - Ligands/non-standard residues: one token per atom

    Example:
        >>> token = Token(1)
        >>> token.value
        1
        >>> token.atom_indices = [1, 2, 3]
        >>> token.centre_atom_index = 2
    """

    def __init__(self, value: int, **kwargs):
        """Initializes a Token.

        Args:
            value: Token value (usually residue type index).
            **kwargs: Additional attributes.
        """
        self.value = value
        self._annot: Dict[str, Any] = {}
        for name, annotation in kwargs.items():
            self._annot[name] = annotation

    def __getattr__(self, attr: str) -> Any:
        """Gets attribute, first checking _annot."""
        if attr in super().__getattribute__("_annot"):
            return self._annot[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def __repr__(self) -> str:
        """String representation."""
        annot_lst = [f"{k}={v}" for k, v in self._annot.items()]
        return f'Token({self.value}, {",".join(annot_lst)})'

    def __setattr__(self, attr: str, value: Any) -> None:
        """Sets attribute, with special handling for _annot and value."""
        if attr == "_annot":
            super().__setattr__(attr, value)
        elif attr == "value":
            super().__setattr__(attr, value)
        else:
            self._annot[attr] = value

    def get_annotation(self, key: str) -> Any:
        """Gets annotation."""
        return self._annot.get(key)

    def set_annotation(self, key: str, value: Any) -> None:
        """Sets annotation."""
        self._annot[key] = value


class TokenArray:
    """Token array for batch operations on a group of Token objects."""

    def __init__(self, tokens: List[Token]):
        """Initializes a TokenArray.

        Args:
            tokens: List of Token objects.
        """
        self.tokens = tokens

    def __repr__(self) -> str:
        """String representation."""
        repr_str = "TokenArray(\n"
        for token in self.tokens:
            repr_str += f"\t{token}\n"
        repr_str += ")"
        return repr_str

    def __len__(self) -> int:
        """Returns the number of tokens."""
        return len(self.tokens)

    def __iter__(self):
        """Iterator."""
        for token in self.tokens:
            yield token

    def __getitem__(self, index):
        """Index access."""
        if isinstance(index, int):
            return self.tokens[index]
        else:
            return TokenArray([self.tokens[i] for i in index])

    def get_annotation(self, category: str) -> List[Any]:
        """Gets an annotation for all tokens.

        Args:
            category: Annotation category.

        Returns:
            List of annotation values for all tokens.
        """
        return [token._annot.get(category) for token in self.tokens]

    def set_annotation(self, category: str, values: List[Any]) -> None:
        """Sets an annotation for all tokens.

        Args:
            category: Annotation category.
            values: List of annotation values, length must equal token count.
        """
        assert len(values) == len(
            self.tokens
        ), f"Length of values ({len(values)}) must match the number of tokens ({len(self.tokens)})"
        for token, value in zip(self.tokens, values):
            token._annot[category] = value

    def get_values(self) -> List[int]:
        """Gets values of all tokens."""
        return [token.value for token in self.tokens]

    def get_atom_indices(self) -> List[List[int]]:
        """Gets atom indices contained in all tokens."""
        return [token.atom_indices for token in self.tokens]

    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary format.

        Returns:
            Dictionary containing all token information.
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
    """AtomArray Tokenizer.

    Tokenizes AtomArray objects into TokenArray.

    Reference: AlphaFold3 SI Chapter 2.6
    - Standard residues (protein, DNA, RNA): one token per residue
    - Ligands and non-standard residues: one token per heavy atom
    """

    def __init__(
        self,
        atom_array: AtomArray,
        std_residues: Optional[Dict[str, int]] = None,
        elems: Optional[Dict[str, int]] = None,
    ):
        """Initializes the Tokenizer.

        Args:
            atom_array: Biotite AtomArray object.
            std_residues: Mapping from standard residues to token values (optional).
            elems: Mapping from elements to token values (optional).
        """
        self.atom_array = atom_array

        # Default standard residue definitions (AlphaFold3 style)
        if std_residues is None:
            from onescience.datapipes.biology.common.features.constants import (
                STD_RESIDUES,
            )
            self.std_residues = STD_RESIDUES
        else:
            self.std_residues = std_residues

        # Default element definitions
        if elems is None:
            from onescience.datapipes.biology.common.features.constants import ELEMS
            self.elems = ELEMS
        else:
            self.elems = elems

    def tokenize(self) -> List[Token]:
        """Tokenizes AtomArray.

        Returns:
            List of Token objects.
        """
        tokens = []
        total_atom_num = 0

        for res in struc.residue_iter(self.atom_array):
            atom_num = len(res)
            first_atom = res[0]
            res_name = first_atom.res_name
            mol_type = getattr(first_atom, "mol_type", "protein")

            # Get token value for standard residues
            res_token = self.std_residues.get(res_name, None)

            if res_token is not None and mol_type != "ligand":
                # Standard residues: one token per residue
                token = Token(res_token)
                atom_indices = list(range(total_atom_num, total_atom_num + atom_num))
                atom_names = [self.atom_array[i].atom_name for i in atom_indices]

                token.atom_indices = atom_indices
                token.atom_names = atom_names
                tokens.append(token)
                total_atom_num += atom_num
            else:
                # Ligands and non-standard residues: one token per atom
                for atom in res:
                    atom_elem = atom.element
                    atom_token = self.elems.get(atom_elem, None)

                    if atom_token is None:
                        # Unknown element, use default value
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
        """Sets token annotations.

        Args:
            token_array: TokenArray object.

        Returns:
            TokenArray with annotations.
        """
        # Get centre atom indices (where centre_atom_mask == 1)
        if hasattr(self.atom_array, "centre_atom_mask"):
            centre_atom_indices = np.where(self.atom_array.centre_atom_mask == 1)[0]
        else:
            # If centre_atom_mask does not exist, use first atom of each residue
            centre_atom_indices = self._get_default_centre_atom_indices()

        token_array.set_annotation("centre_atom_index", centre_atom_indices.tolist())
        assert len(token_array) == len(centre_atom_indices), \
            f"Token count mismatch: {len(token_array)} vs {len(centre_atom_indices)}"

        return token_array

    def _get_default_centre_atom_indices(self) -> np.ndarray:
        """Gets default centre atom indices.

        Used when centre_atom_mask does not exist:
        - Protein: CA atom
        - DNA/RNA: C1' atom
        - Others: first atom

        Returns:
            Array of centre atom indices.
        """
        centre_indices = []

        for res in struc.residue_iter(self.atom_array):
            res_atoms = res
            atom_names = [atom.atom_name for atom in res_atoms]
            global_start_idx = next(iter(res_atoms)).array_index(0)

            # Find centre atom
            if "CA" in atom_names:
                idx = atom_names.index("CA")
            elif "C1'" in atom_names:
                idx = atom_names.index("C1'")
            else:
                idx = 0

            centre_indices.append(global_start_idx + idx)

        return np.array(centre_indices, dtype=np.int64)

    def get_token_array(self) -> TokenArray:
        """Gets TokenArray with annotations.

        Returns:
            TokenArray containing annotations such as atom_indices, centre_atom_index.

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
    """Convenience function: creates TokenArray from AtomArray.

    Args:
        atom_array: Biotite AtomArray object.
        use_centre_atom_mask: Whether to use centre_atom_mask (if it exists).

    Returns:
        TokenArray object.
    """
    tokenizer = AtomArrayTokenizer(atom_array)
    return tokenizer.get_token_array()


def token_array_to_atom_indices(token_array: TokenArray) -> List[int]:
    """Gets centre atom indices for all tokens in TokenArray.

    Args:
        token_array: TokenArray object.

    Returns:
        List of centre atom indices.
    """
    return token_array.get_annotation("centre_atom_index")


def get_token_atom_mapping(token_array: TokenArray, num_atoms: int) -> np.ndarray:
    """Gets the atom-to-token mapping array.

    Args:
        token_array: TokenArray object.
        num_atoms: Total number of atoms.

    Returns:
        Atom-to-token mapping array, shape: [N_atom], values are token indices.
    """
    atom_to_token = np.full(num_atoms, -1, dtype=np.int64)

    for token_idx, token in enumerate(token_array):
        for atom_idx in token.atom_indices:
            if 0 <= atom_idx < num_atoms:
                atom_to_token[atom_idx] = token_idx

    return atom_to_token
