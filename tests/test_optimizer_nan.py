"""Single optimizer steps on a tiny CPU model do not produce NaNs."""

import torch

from tiny_swap_bench.config_schema import ModelConfig, TrainConfig
from tiny_swap_bench.model.reference_transformer import ReferenceTransformer
from tiny_swap_bench.optim import build_optimizer


def _one_step(optim_name: str):
    cfg_m = ModelConfig(
        n_layer=1,
        n_embd=32,
        n_head=4,
        seq_len=16,
        vocab_size=128,
        pe="nope",
        norm="pre_ln_ln",
    )
    tc = TrainConfig(
        lr=1e-3,
        optimizer=optim_name,  # type: ignore[arg-type]
        max_train_tokens=1000,
        warmup_steps=1,
    )
    m = ReferenceTransformer(cfg_m)
    opt = build_optimizer(m, tc)
    x = torch.randint(0, cfg_m.vocab_size, (2, cfg_m.seq_len), dtype=torch.long)
    logits = m(x)
    loss = logits.float().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        ps = torch.stack([p.float().mean() for p in m.parameters()])
    assert torch.isfinite(ps).all()


def test_adamw_step():
    _one_step("adamw")


def test_lion_step():
    _one_step("lion")


def test_muon_step():
    _one_step("muon")
