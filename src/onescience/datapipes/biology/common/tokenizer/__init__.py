"""
通用Token化模块

参考AlphaFold3 SI Chapter 2.6实现
将AtomArray转换为TokenArray，供多种模型使用
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
