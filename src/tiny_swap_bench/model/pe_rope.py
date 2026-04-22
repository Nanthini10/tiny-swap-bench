"""Rotary positional embeddings (apply inside attention to Q/K)."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., 2*d_half) -> rotate pairs
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, T, d_h); cos, sin: (T, d_h)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_h)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (rotate_half(q) * sin)  # (B, H, T, d_h)
    k_out = (k * cos) + (rotate_half(k) * sin)  # (B, H, T, d_h)
    return q_out, k_out


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for positions ``0 .. seq_len-1``."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (T,)
    freqs = torch.outer(t, inv_freq)  # (T, d_h/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, d_h)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin  # each (T, d_h)
