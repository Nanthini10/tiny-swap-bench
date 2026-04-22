"""Parameter counting with explicit non-embedding definition."""

from __future__ import annotations

import torch.nn as nn


def count_non_embedding_params(model: nn.Module) -> int:
    """Sum trainable params excluding ``wte`` and learned absolute ``wpe`` (if present)."""
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "wte.weight":
            continue
        if name.endswith("wpe.weight") or ".wpe." in name:
            continue
        total += p.numel()
    return int(total)


def count_total_trainable(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
