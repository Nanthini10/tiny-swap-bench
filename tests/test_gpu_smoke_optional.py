"""Full forward + backward on CUDA if available (skipped otherwise)."""

import pytest
import torch

from tiny_swap_bench.config_schema import ModelConfig
from tiny_swap_bench.model.reference_transformer import ReferenceTransformer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_forward_backward_one_step():
    cfg = ModelConfig(n_layer=2, n_embd=64, n_head=4, seq_len=64, vocab_size=1000, pe="learned_abs", norm="pre_ln_ln")
    m = ReferenceTransformer(cfg).cuda().bfloat16()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), device="cuda", dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = m(x)
        loss = logits.float().mean()
    loss.backward()
    assert torch.isfinite(loss)
