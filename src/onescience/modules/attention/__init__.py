
from importlib import import_module


_ESM_ATTENTION_EXPORTS = {
    "ColumnSelfAttention": ("onescience.modules.attention.esm_axial_attention", "ColumnSelfAttention"),
    "MultiheadAttention": ("onescience.modules.attention.esm_multihead_attention", "MultiheadAttention"),
    "RotaryEmbedding": ("onescience.modules.attention.esm_rotary_embedding", "RotaryEmbedding"),
    "RowSelfAttention": ("onescience.modules.attention.esm_axial_attention", "RowSelfAttention"),
}

__all__ = [
    "ColumnSelfAttention",
    "MultiheadAttention",
    "RotaryEmbedding",
    "RowSelfAttention",
]


def __getattr__(name):
    if name not in _ESM_ATTENTION_EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = _ESM_ATTENTION_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
