"""Inference 推理工具模块"""

import torch
import torch.nn.functional as F


class KVCache:
    """键值缓存"""
    def __init__(self):
        self.k_cache = []
        self.v_cache = []

    def update(self, k, v):
        self.k_cache.append(k)
        self.v_cache.append(v)
        return torch.cat(self.k_cache, dim=2), torch.cat(self.v_cache, dim=2)

    def clear(self):
        self.k_cache.clear()
        self.v_cache.clear()


def top_k_sampling(logits, k=50, temperature=1.0):
    """Top-K采样"""
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return top_k_indices.gather(-1, next_token)


def top_p_sampling(logits, p=0.9, temperature=1.0):
    """Top-P (Nucleus) 采样"""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return sorted_indices.gather(-1, next_token)


def test_inference():
    print("\n[测试] Inference")

    cache = KVCache()
    k = torch.randn(2, 4, 10, 32)
    v = torch.randn(2, 4, 10, 32)
    k_cat, v_cat = cache.update(k, v)
    assert k_cat.shape == (2, 4, 10, 32)

    logits = torch.randn(1, 1000)
    token_k = top_k_sampling(logits, k=50)
    token_p = top_p_sampling(logits, p=0.9)
    assert token_k.shape == (1, 1)
    assert token_p.shape == (1, 1)
    print("✓ KVCache & Sampling")


if __name__ == "__main__":
    test_inference()
