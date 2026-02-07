import torch

from torch import nn

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, t):

        hidden_size = self.config.hidden_size
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        x = self.timestep_embedding(t)
        x = nn.Dense(hidden_size, kernel_init=normal(0.02), dtype=arr_dtype)(x)
        x = nn.silu(x)
        x = nn.Dense(hidden_size, kernel_init=normal(0.02), dtype=arr_dtype)(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.config.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1) ### TODO: pi here?
        return embedding