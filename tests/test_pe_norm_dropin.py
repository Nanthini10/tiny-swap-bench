"""PE and norm variants preserve I/O shapes."""

import torch

from tiny_swap_bench.config_schema import ModelConfig
from tiny_swap_bench.model.reference_transformer import ReferenceTransformer


def _run_variant(pe: str, norm: str):
    cfg = ModelConfig(
        n_layer=2,
        n_embd=64,
        n_head=4,
        seq_len=32,
        vocab_size=512,
        pe=pe,  # type: ignore[arg-type]
        norm=norm,  # type: ignore[arg-type]
    )
    m = ReferenceTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), dtype=torch.long)
    y = m(x)
    assert y.shape == (2, cfg.seq_len, cfg.vocab_size)


def test_pe_variants_shapes():
    for pe in ("learned_abs", "rope", "nope", "alibi"):
        _run_variant(pe, "pre_ln_ln")


def test_norm_variants_shapes():
    for norm in ("pre_ln_ln", "pre_ln_rms", "post_ln_ln", "pre_ln_rms_qk"):
        _run_variant("nope", norm)
