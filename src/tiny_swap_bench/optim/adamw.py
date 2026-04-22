"""AdamW (torch built-in wrapper for consistent factory)."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW

from tiny_swap_bench.config_schema import TrainConfig


def build_adamw(params: list[nn.Parameter], cfg: TrainConfig) -> AdamW:
    return AdamW(
        params,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
