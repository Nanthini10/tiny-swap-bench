"""Lion optimizer — decoupled weight decay, momentum sign update."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim.optimizer import Optimizer

from tiny_swap_bench.config_schema import TrainConfig


class Lion(Optimizer):
    """Lion (Chen et al.): single momentum with sign-based update."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
    ) -> None:
        _ = betas[1]
        defaults = dict(lr=lr, beta1=betas[0], weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        if closure is not None:
            raise NotImplementedError("Lion does not support closure in this scaffold.")
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                update = torch.sign(m)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)


def build_lion(params: list, cfg: TrainConfig) -> Lion:
    return Lion(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
