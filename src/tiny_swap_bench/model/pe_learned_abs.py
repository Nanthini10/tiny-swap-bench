"""Learned absolute positional embeddings (additive to token embeddings)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedAbsolutePE(nn.Module):
    """Adds learned position embeddings for positions ``0 .. max_pos-1``."""

    def __init__(self, max_pos: int, n_embd: int) -> None:
        super().__init__()
        self.wpe = nn.Embedding(max_pos, n_embd)  # (max_pos, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, _d = x.shape
        pos = torch.arange(t, device=x.device, dtype=torch.long)  # (T,)
        pos_emb = self.wpe(pos)  # (T, D)
        return x + pos_emb.unsqueeze(0).expand(b, -1, -1)  # (B, T, D)
