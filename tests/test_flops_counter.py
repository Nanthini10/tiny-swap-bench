"""Analytical FLOPs vs hand derivation for a 2-layer toy model (within 2%)."""

from tiny_swap_bench.config_schema import ModelConfig
from tiny_swap_bench.eval.flops_count import forward_flops_per_forward


def test_flops_matches_hand_toy():
    l, d, h = 2, 64, 4
    t, b, v = 8, 2, 100
    inner = 4 * d
    dh = d // h
    cfg = ModelConfig(n_layer=l, n_embd=d, n_head=h, mlp_ratio=4.0, seq_len=t, vocab_size=v, pe="nope", norm="pre_ln_ln")

    coded = forward_flops_per_forward(cfg, batch=b, seq_len=t)

    per_layer_matmul = (
        2 * t * d * (3 * d)  # QKV
        + 2 * t * d * d  # out
        + 2 * t * d * inner  # up
        + 2 * t * inner * d  # down
    )
    qk = 2 * h * (t * t) * dh
    pv = 2 * h * (t * t) * dh
    per_layer_attn = qk + pv
    per_layer_norm = 2 * t * (6 * d)
    final_norm = t * (6 * d)
    lm_head = 2 * b * t * d * v
    hand = l * (per_layer_matmul + per_layer_attn + per_layer_norm) + final_norm + lm_head

    rel = abs(float(coded) - float(hand)) / max(float(hand), 1.0)
    assert rel <= 0.02, f"coded={coded} hand={hand} rel_err={rel}"
