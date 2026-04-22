"""No positional encoding (identity)."""

from __future__ import annotations

import torch
import torch.nn as nn


class NoPE(nn.Module):
    """Pass-through; model relies on causal mask and content only."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x
