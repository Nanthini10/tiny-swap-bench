"""Reference transformer output shapes on random tokens."""

import torch

from tiny_swap_bench.config_schema import ModelConfig
from tiny_swap_bench.model.reference_transformer import ReferenceTransformer


def test_reference_logits_shape_cpu():
    cfg = ModelConfig(n_layer=2, n_embd=64, n_head=4, seq_len=32, vocab_size=512, pe="learned_abs", norm="pre_ln_ln")
    m = ReferenceTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), dtype=torch.long)
    logits = m(x)  # (B, T, V)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
