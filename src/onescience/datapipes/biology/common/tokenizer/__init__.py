"""Module for general tokenization.

Implements tokenization following AlphaFold3 SI Chapter 2.6.
Converts AtomArray to TokenArray for use by various models.
"""

from onescience.datapipes.biology.common.tokenizer.tokenizer import (
    Token,
    TokenArray,
    AtomArrayTokenizer,
    create_token_array,
    token_array_to_atom_indices,
    get_token_atom_mapping,
)

__all__ = [
    "Token",
    "TokenArray",
    "AtomArrayTokenizer",
    "create_token_array",
    "token_array_to_atom_indices",
    "get_token_atom_mapping",
]
