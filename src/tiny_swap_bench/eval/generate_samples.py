"""Greedy/sampling generation for qualitative + judge scaffolding."""

from __future__ import annotations

import torch


def _truncate_context_for_forward(model: torch.nn.Module, idx: torch.Tensor) -> torch.Tensor:
    """Learned absolute PE only supports positions ``0 .. seq_len-1``; keep the last ``seq_len`` tokens."""
    cfg = getattr(model, "cfg", None)
    if cfg is None or getattr(cfg, "pe", None) != "learned_abs":
        return idx
    limit = int(cfg.seq_len)
    return idx if idx.size(1) <= limit else idx[:, -limit:]


@torch.no_grad()
def generate_completion(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressive continuation; ``prompt_ids`` shape ``(1, T0)``, returns ``(1, T0+N)``."""
    idx = prompt_ids.to(device)  # (1, T0)
    model.eval()
    amp_on = dtype == torch.bfloat16 and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_on):
        for _ in range(max_new_tokens):
            ctx = _truncate_context_for_forward(model, idx)
            logits = model(ctx)  # (1, T_ctx, V)
            logits_step = logits[:, -1, :] / max(temperature, 1e-6)  # (1, V)
            if top_k is not None:
                v, _ = torch.topk(logits_step, top_k)
                logits_step = logits_step.masked_fill(logits_step < v[:, [-1]], float("-inf"))
            probs = torch.softmax(logits_step, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx = torch.cat([idx, next_id], dim=1)
    return idx
