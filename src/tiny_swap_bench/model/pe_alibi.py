"""ALiBi linear biases for causal attention (fixed slopes per head)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_alibi_slopes(n_heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Slopes ``(H,)`` following ALiBi (Press et al.)."""
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes = torch.pow(2.0, -torch.arange(0, closest_power_of_2, device=device, dtype=torch.float32))
    if closest_power_of_2 != n_heads:
        extra = torch.pow(
            2.0,
            -torch.arange(1, 2 * (n_heads - closest_power_of_2) + 1, 2, device=device, dtype=torch.float32),
        )
        slopes = torch.cat([slopes, extra], dim=0)
    slopes = slopes[:n_heads]
    return slopes.to(dtype=dtype)


def alibi_attention_bias(seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
    """Causal ALiBi bias ``(H, T, T)`` added to attention logits before softmax.

    For query row ``i`` and key column ``j`` with ``j <= i``, bias is ``-m * (i - j)``.
    """
    # slopes: (H,)
    i = torch.arange(seq_len, device=slopes.device, dtype=slopes.dtype).unsqueeze(1)  # (T, 1)
    j = torch.arange(seq_len, device=slopes.device, dtype=slopes.dtype).unsqueeze(0)  # (1, T)
    diff = i - j  # (T, T)
    bias = -slopes.unsqueeze(-1).unsqueeze(-1) * diff.unsqueeze(0)  # (H, T, T)
    return bias


class AlibiBias(nn.Module):
    """Buffers slopes; forward builds ``(H, T, T)`` bias for current ``T``."""

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        slopes = build_alibi_slopes(n_heads, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)  # (H,)

    def forward(self, seq_len: int) -> torch.Tensor:
        return alibi_attention_bias(seq_len, self.slopes.to(dtype=torch.float32))  # (H, T, T)
