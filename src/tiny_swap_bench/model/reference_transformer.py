"""Decoder-only transformer (reference architecture + swap hooks)."""

from __future__ import annotations

import torch
import torch.nn as nn

from tiny_swap_bench.config_schema import ModelConfig, NormName, PeName
from tiny_swap_bench.model.decoder_block import PostLNBlock, PreLNBlock
from tiny_swap_bench.model.norm_layer_ln import layer_norm
from tiny_swap_bench.model.norm_layer_rms import RMSNorm
from tiny_swap_bench.model.pe_learned_abs import LearnedAbsolutePE
from tiny_swap_bench.model.pe_nope import NoPE


def _make_norm(n_embd: int, use_rms: bool) -> nn.Module:
    return RMSNorm(n_embd) if use_rms else layer_norm(n_embd)


def build_blocks(cfg: ModelConfig) -> tuple[nn.ModuleList, nn.Module]:
    """Return ``(blocks, ln_f)`` for the given norm layout."""
    norm: NormName = cfg.norm
    pe: PeName = cfg.pe
    use_rms = norm in ("pre_ln_rms", "pre_ln_rms_qk")
    use_qk = norm == "pre_ln_rms_qk"
    is_post = norm == "post_ln_ln"

    blocks: list[nn.Module] = []
    for _ in range(cfg.n_layer):
        ln1 = _make_norm(cfg.n_embd, use_rms and not is_post)
        ln2 = _make_norm(cfg.n_embd, use_rms and not is_post)
        if is_post:
            ln1 = layer_norm(cfg.n_embd)
            ln2 = layer_norm(cfg.n_embd)
        if is_post:
            blk = PostLNBlock(
                cfg.n_embd,
                cfg.n_head,
                cfg.mlp_ratio,
                cfg.dropout,
                pe,
                use_qk,
                cfg.seq_len,
                ln1,
                ln2,
            )
        else:
            blk = PreLNBlock(
                cfg.n_embd,
                cfg.n_head,
                cfg.mlp_ratio,
                cfg.dropout,
                pe,
                use_qk,
                cfg.seq_len,
                ln1,
                ln2,
            )
        blocks.append(blk)

    ln_f = _make_norm(cfg.n_embd, use_rms and not is_post)
    if is_post:
        ln_f = layer_norm(cfg.n_embd)
    return nn.ModuleList(blocks), ln_f


def build_positional(cfg: ModelConfig) -> nn.Module:
    pe: PeName = cfg.pe
    if pe == "learned_abs":
        return LearnedAbsolutePE(cfg.seq_len, cfg.n_embd)
    return NoPE()


class ReferenceTransformer(nn.Module):
    """Token embeddings + optional learned PE + stack of decoder blocks + final norm + LM head tied to wte."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # (V, D)
        self.drop = nn.Dropout(cfg.dropout)
        self.pos = build_positional(cfg)
        self.blocks, self.ln_f = build_blocks(cfg)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) int64 token ids
        _b, t = idx.shape
        x = self.wte(idx)  # (B, T, D)
        x = self.drop(x)
        x = self.pos(x)  # (B, T, D)
        for block in self.blocks:
            x = block(x)  # (B, T, D)
        x = self.ln_f(x)  # (B, T, D)
        logits = x @ self.wte.weight.T  # (B, T, V) tied head
        return logits
