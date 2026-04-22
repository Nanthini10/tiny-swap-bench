"""Optimizer factories."""

from __future__ import annotations

import torch.nn as nn

from tiny_swap_bench.config_schema import TrainConfig
from tiny_swap_bench.optim.adamw import build_adamw
from tiny_swap_bench.optim.lion import build_lion
from tiny_swap_bench.optim.muon import MuonHybrid, build_muon


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    name = cfg.optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return build_adamw(params, cfg)
    if name == "lion":
        return build_lion(params, cfg)
    if name == "muon":
        return build_muon(model, cfg)
    raise NotImplementedError(f"Unknown optimizer {name!r}.")
