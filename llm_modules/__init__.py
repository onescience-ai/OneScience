"""
LLM通用组件库 - 模块化版本
"""

from .embedding import *
from .attention import *
from .feedforward import *
from .normalization import *
from .transformer import *
from .moe import *
from .peft import *
from .rag import *
from .agent import *
from .vlm import *
from .quantization import *
from .sparse import *
from .inference import *
from .mamba import *
from .rnn import *

__all__ = [
    # Embedding
    'TokenEmbedding', 'RoPE', 'ALiBi', 'LearnedPositionalEmbedding',
    # Attention
    'MultiHeadAttention', 'GroupedQueryAttention', 'SlidingWindowAttention', 'FlashAttention',
    # Feedforward
    'SwiGLU', 'GeGLU', 'GLU', 'Mish',
    # Normalization
    'RMSNorm', 'LayerNorm', 'GroupNorm',
    # Transformer
    'TransformerBlock',
    # MoE
    'MoE', 'MixtureOfAdapters', 'RAMoLE',
    # PEFT
    'LoRALinear', 'QLoRALinear', 'Adapter', 'PrefixTuning', 'PromptTuning',
    # RAG
    'CrossAttention', 'RAGLayer',
    # Agent
    'ToolCallingHead', 'ReflectionLayer',
    # VLM
    'VisionEncoder', 'MultiModalFusion',
    # Quantization
    'Int8Linear', 'BinaryLinear',
    # Sparse
    'TopKGating', 'ExpertChoice',
    # Inference
    'KVCache', 'top_k_sampling', 'top_p_sampling',
    # Mamba
    'S6', 'MambaBlock', 'Mamba',
    # RNN
    'LSTM', 'GRU', 'BiLSTM', 'RWKV',
]
