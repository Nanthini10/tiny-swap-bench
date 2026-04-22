"""Validation cross-entropy on packed token batches."""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn.functional as F

from tiny_swap_bench.data.tinystories import Batch


@torch.no_grad()
def batches_loss(
    model: torch.nn.Module,
    batches: Iterator[Batch],
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
    amp_enabled: bool,
) -> tuple[float, int]:
    """Average CE over up to ``max_batches`` batches; returns (mean_loss, batches_used)."""
    losses: list[float] = []
    used = 0
    for batch in batches:
        if used >= max_batches:
            break
        x = batch.input_ids  # (B, T)
        y = batch.labels  # (B, T)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            logits = model(x)  # (B, T, V)
        logits_flat = logits.view(-1, logits.size(-1))  # (B*T, V)
        y_flat = y.view(-1)
        loss = F.cross_entropy(logits_flat, y_flat)  # scalar
        losses.append(loss.item())
        used += 1
    if not losses:
        return float("nan"), 0
    mean = sum(losses) / len(losses)
    return mean, used


def perplexity(loss_natural: float) -> float:
    return math.exp(loss_natural)
