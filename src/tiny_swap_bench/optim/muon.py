"""Muon-style optimizer: Newton–Schulz orthogonalized updates on 2D weights + AdamW elsewhere."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from tiny_swap_bench.config_schema import TrainConfig


@torch.no_grad()
def zeropower_via_newtonschulz5(g: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """Orthogonalize gradient matrix via Newton–Schulz iteration."""
    assert g.ndim == 2
    if g.size(0) > g.size(1):
        return zeropower_via_newtonschulz5(g.T, ns_steps).T
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    norm = x.norm()
    x = x / (norm + 1e-7)
    for _ in range(ns_steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + (b_mat @ x)
    return x


def split_muon_adam_params(model: nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon_ps: list[torch.nn.Parameter] = []
    adam_ps: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "wte.weight":
            adam_ps.append(p)
            continue
        if p.ndim == 2:
            muon_ps.append(p)
        else:
            adam_ps.append(p)
    return muon_ps, adam_ps


class MuonHybrid:
    """Compatible with training loop: ``zero_grad()`` / ``step()`` / checkpointing."""

    def __init__(self, model: nn.Module, cfg: TrainConfig) -> None:
        mu_p, ad_p = split_muon_adam_params(model)
        self.mu_params = mu_p
        self.lr_muon = cfg.lr
        self.momentum = cfg.beta1
        self.state: dict[int, dict[str, torch.Tensor]] = {}
        self.adam = AdamW(
            ad_p,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

    def state_dict(self) -> dict[str, Any]:
        bufs = []
        for p in self.mu_params:
            pid = id(p)
            buf = self.state.get(pid, {}).get("buf", torch.zeros_like(p))
            bufs.append(buf.detach().cpu())
        return {"adam": self.adam.state_dict(), "muon_bufs": bufs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.adam.load_state_dict(state["adam"])
        bufs = state.get("muon_bufs", [])
        self.state = {}
        for p, b in zip(self.mu_params, bufs):
            self.state[id(p)] = {"buf": b.to(device=p.device, dtype=p.dtype)}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adam.zero_grad(set_to_none=set_to_none)
        for p in self.mu_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        if closure is not None:
            raise NotImplementedError("MuonHybrid does not support closure.")
        self.adam.step()
        for p in self.mu_params:
            if p.grad is None:
                continue
            g = p.grad
            assert g.ndim == 2
            pid = id(p)
            if pid not in self.state:
                self.state[pid] = {"buf": torch.zeros_like(g)}
            buf = self.state[pid]["buf"]
            buf.mul_(self.momentum).add_(g, alpha=1.0 - self.momentum)
            upd = zeropower_via_newtonschulz5(buf).to(dtype=p.dtype, device=p.device)
            scale = math.sqrt(max(p.shape[0], p.shape[1]))
            p.add_(upd, alpha=-self.lr_muon * scale)


def build_muon(model: nn.Module, cfg: TrainConfig) -> MuonHybrid:
    return MuonHybrid(model, cfg)
