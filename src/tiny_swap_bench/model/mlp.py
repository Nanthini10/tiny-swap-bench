"""Position-wise feed-forward (GELU, 4× expansion by default)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_embd: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner = int(mlp_ratio * n_embd)
        self.fc = nn.Linear(n_embd, inner)  # (B, T, D) -> (B, T, inner)
        self.proj = nn.Linear(inner, n_embd)  # (B, T, inner) -> (B, T, D)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.act(self.fc(x))  # (B, T, inner)
        return self.proj(h)  # (B, T, D)
