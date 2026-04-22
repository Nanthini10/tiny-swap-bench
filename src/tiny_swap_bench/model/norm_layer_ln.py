"""LayerNorm factory matching GPT-style epsilon."""

from __future__ import annotations

import torch.nn as nn


def layer_norm(n_embd: int) -> nn.LayerNorm:
    return nn.LayerNorm(n_embd, eps=1e-5)
