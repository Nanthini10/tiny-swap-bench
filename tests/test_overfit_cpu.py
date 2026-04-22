"""Overfit a single tiny batch (CPU, full precision)."""

import torch
import torch.nn.functional as F

from tiny_swap_bench.config_schema import ModelConfig, TrainConfig
from tiny_swap_bench.model.reference_transformer import ReferenceTransformer
from tiny_swap_bench.optim import build_optimizer


def test_overfit_one_batch_under_200_steps():
    cfg = ModelConfig(
        n_layer=2,
        n_embd=64,
        n_head=4,
        seq_len=32,
        vocab_size=256,
        pe="learned_abs",
        norm="pre_ln_ln",
        dropout=0.0,
    )
    train_cfg = TrainConfig(
        lr=3e-3,
        warmup_steps=1,
        batch_size=4,
        grad_accum_steps=1,
        max_train_tokens=1_000_000,
        optimizer="adamw",
        seed=0,
    )
    model = ReferenceTransformer(cfg)
    opt = build_optimizer(model, train_cfg)

    torch.manual_seed(0)
    buf = torch.randint(0, cfg.vocab_size, (cfg.seq_len + 1,))
    x = buf[: cfg.seq_len].repeat(4, 1)
    y = buf[1 : cfg.seq_len + 1].repeat(4, 1)

    loss_val = float("inf")
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        loss = F.cross_entropy(logits_flat, y_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = float(loss.item())
        if loss_val < 0.05:
            break

    assert loss_val < 0.1, f"failed to overfit, final loss={loss_val}"
