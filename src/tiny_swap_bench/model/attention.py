"""Causal multi-head self-attention with optional RoPE, ALiBi, QK-norm."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_swap_bench.model.pe_alibi import AlibiBias
from tiny_swap_bench.model.pe_rope import apply_rope, build_rope_cache
from tiny_swap_bench.model.norm_layer_rms import RMSNorm


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        pe: str,
        use_qk_norm: bool,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.pe = pe
        self.use_qk_norm = use_qk_norm
        self.max_seq_len = max_seq_len

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # (B,T,D) -> (B,T,3D)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.qk_norm_q = RMSNorm(self.head_dim) if use_qk_norm else None
        self.qk_norm_k = RMSNorm(self.head_dim) if use_qk_norm else None

        self.alibi = AlibiBias(n_head) if pe == "alibi" else None

        if pe == "rope":
            cos, sin = build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"), torch.float32)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.register_buffer("rope_cos", torch.empty(0), persistent=False)
            self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, d = x.shape
        qkv = self.c_attn(x)  # (B, T, 3D)
        q, k, v = qkv.split(self.n_embd, dim=-1)  # each (B, T, D)
        qh = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d_h)
        kh = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        vh = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d_h)

        if self.use_qk_norm:
            assert self.qk_norm_q is not None and self.qk_norm_k is not None
            qh = self.qk_norm_q(qh)  # (B, H, T, d_h)
            kh = self.qk_norm_k(kh)

        if self.pe == "rope":
            cos = self.rope_cos[:t].to(dtype=x.dtype, device=x.device)  # (T, d_h)
            sin = self.rope_sin[:t].to(dtype=x.dtype, device=x.device)
            qh, kh = apply_rope(qh, kh, cos, sin)  # (B, H, T, d_h)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = (qh @ kh.transpose(-2, -1)) * scale  # (B, H, T, T)

        if self.pe == "alibi" and self.alibi is not None:
            bias = self.alibi(t).to(dtype=att.dtype, device=att.device)  # (H, T, T)
            att = att + bias.unsqueeze(0)

        causal = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ vh  # (B, H, T, d_h)
        y = y.transpose(1, 2).contiguous().view(b, t, d)  # (B, T, D)
        y = self.c_proj(y)  # (B, T, D)
        return self.resid_drop(y)
