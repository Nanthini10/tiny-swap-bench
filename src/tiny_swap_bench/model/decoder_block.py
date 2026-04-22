"""Pre-LN and post-LN decoder blocks with pluggable norms."""

from __future__ import annotations

import torch
import torch.nn as nn

from tiny_swap_bench.model.attention import CausalSelfAttention
from tiny_swap_bench.model.mlp import MLP


class PreLNBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        mlp_ratio: float,
        dropout: float,
        pe: str,
        use_qk_norm: bool,
        max_seq_len: int,
        ln1: nn.Module,
        ln2: nn.Module,
    ) -> None:
        super().__init__()
        self.ln1 = ln1
        self.ln2 = ln2
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, pe, use_qk_norm, max_seq_len)
        self.mlp = MLP(n_embd, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.attn(self.ln1(x))  # (B, T, D)
        x = x + self.mlp(self.ln2(x))  # (B, T, D)
        return x


class PostLNBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        mlp_ratio: float,
        dropout: float,
        pe: str,
        use_qk_norm: bool,
        max_seq_len: int,
        ln1: nn.Module,
        ln2: nn.Module,
    ) -> None:
        super().__init__()
        self.ln1 = ln1
        self.ln2 = ln2
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, pe, use_qk_norm, max_seq_len)
        self.mlp = MLP(n_embd, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.ln1(x + self.attn(x))  # (B, T, D)
        x = self.ln2(x + self.mlp(x))  # (B, T, D)
        return x
