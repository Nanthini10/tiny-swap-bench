"""Analytical forward FLOP estimates (matmul convention: 2MNK FLOPs)."""

from __future__ import annotations

from dataclasses import dataclass

from tiny_swap_bench.config_schema import ModelConfig


@dataclass(frozen=True)
class FlopBreakdown:
    per_layer_matmul: int
    per_layer_attn_qk_softmax_v: int
    per_layer_norm_approx: int
    lm_head_tied: int
    final_norm_approx: int


def forward_flops_per_forward(cfg: ModelConfig, *, batch: int, seq_len: int) -> int:
    """Total forward FLOPs for one micro-batch forward to logits (including tied LM head)."""
    t = seq_len
    d = cfg.n_embd
    h = cfg.n_head
    dh = d // h
    l = cfg.n_layer
    v = cfg.vocab_size
    inner = int(cfg.mlp_ratio * d)

    # Attention projections + MLP (matmuls)
    attn_proj = 2 * t * d * (3 * d) + 2 * t * d * d  # QKV + out
    mlp_fc = 2 * t * d * inner + 2 * t * inner * d  # up + down
    per_layer_matmul = attn_proj + mlp_fc

    # QK^T and softmax-weighted V per layer
    qk = 2 * h * (t * t) * dh
    pv = 2 * h * (t * t) * dh
    per_layer_attn = qk + pv

    # Small explicit norm terms (~6·D elementwise ops per norm application per token)
    per_layer_norm = 2 * t * (6 * d)  # two norms per block
    final_norm = t * (6 * d)

    # Tied LM head: (B,T,D) @ (D,V)
    lm_head = 2 * batch * t * d * v

    total = l * (per_layer_matmul + per_layer_attn + per_layer_norm) + final_norm + lm_head
    return int(total)


def forward_flops_scalar_estimate(cfg: ModelConfig, *, batch: int, seq_len: int) -> FlopBreakdown:
    """Structured view (debug / tests)."""
    t = seq_len
    d = cfg.n_embd
    h = cfg.n_head
    dh = d // h
    l = cfg.n_layer
    v = cfg.vocab_size
    inner = int(cfg.mlp_ratio * d)

    attn_proj = 2 * t * d * (3 * d) + 2 * t * d * d
    mlp_fc = 2 * t * d * inner + 2 * t * inner * d
    per_layer_matmul = attn_proj + mlp_fc
    qk = 2 * h * (t * t) * dh
    pv = 2 * h * (t * t) * dh
    per_layer_attn = qk + pv
    per_layer_norm = 2 * t * (6 * d)
    final_norm = t * (6 * d)
    lm_head = 2 * batch * t * d * v
    return FlopBreakdown(
        per_layer_matmul=int(per_layer_matmul),
        per_layer_attn_qk_softmax_v=int(per_layer_attn),
        per_layer_norm_approx=int(per_layer_norm),
        lm_head_tied=int(lm_head),
        final_norm_approx=int(final_norm),
    )


def training_flops_from_forward(forward_flops: int) -> int:
    """Training step convention: ``3 × forward`` (forward + ~2× backward)."""
    return int(3 * forward_flops)
