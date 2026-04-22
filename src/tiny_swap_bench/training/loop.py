"""Single-file training loop (Karpathy-style, top-to-bottom).

Training requires CUDA — CPU fallback is forbidden by project rules.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tiny_swap_bench.config_schema import RunConfig, runconfig_to_dict
from tiny_swap_bench.data.tinystories import (
    TinyStoriesTokenizer,
    build_batches,
    load_train_val_rows,
    make_train_token_iterator,
)
from tiny_swap_bench.eval.flops_count import forward_flops_per_forward, training_flops_from_forward
from tiny_swap_bench.eval.generate_samples import generate_completion
from tiny_swap_bench.eval.llm_judge_stub import aggregate_judge_scores, load_prompts_and_rubric, score_completion_with_env
from tiny_swap_bench.eval.validation_loss import batches_loss
from tiny_swap_bench.model.reference_transformer import ReferenceTransformer
from tiny_swap_bench.optim import build_optimizer
from tiny_swap_bench.optim.muon import MuonHybrid
from tiny_swap_bench.utils.param_counts import count_non_embedding_params, count_total_trainable
from tiny_swap_bench.utils.run_metadata import write_run_metadata
from tiny_swap_bench.utils.seeding import set_seed


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; refusing to fall back to CPU.")
    return torch.device("cuda", index=0)


def cosine_lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps <= 0:
        warmup_steps = 1
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def optimizer_step_lr(opt: Any, base_lr: float, mult: float) -> None:
    lr = base_lr * mult
    if isinstance(opt, MuonHybrid):
        for g in opt.adam.param_groups:
            g["lr"] = lr
        opt.lr_muon = lr
        return
    for g in opt.param_groups:
        g["lr"] = lr


def pack_val_tokens(val_ds, tok: TinyStoriesTokenizer) -> list[int]:
    ids: list[int] = []
    for i in range(len(val_ds)):
        ids.extend(tok.encode_text(val_ds[i]["text"]))
    return ids


def train_run(
    cfg: RunConfig,
    run_dir: Path,
    *,
    prompts_path: Path | None = None,
    smoke: bool = False,
    smoke_eval_batches: int = 2,
) -> dict[str, Any]:
    device = require_cuda()
    set_seed(cfg.train.seed, cudnn_deterministic=False)
    random.seed(cfg.train.seed)

    run_dir = Path(run_dir)
    write_run_metadata(run_dir, cfg)

    dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float32

    tok = TinyStoriesTokenizer(cfg.data.tokenizer_name)
    train_ds, val_ds = load_train_val_rows(cfg.data.dataset_path, cfg.data.dataset_split_train, cfg.data.val_fraction)

    model = ReferenceTransformer(cfg.model).to(device=device)
    non_emb = count_non_embedding_params(model)
    total_p = count_total_trainable(model)

    opt = build_optimizer(model, cfg.train)

    seq_len = cfg.model.seq_len
    batch_size = cfg.train.batch_size
    accum = max(1, cfg.train.grad_accum_steps)
    tokens_per_step = batch_size * seq_len * accum

    max_tokens = cfg.train.max_train_tokens
    if smoke:
        max_tokens = min(max_tokens, 8192)

    total_steps_est = max(1, math.ceil(max_tokens / tokens_per_step))

    train_tokens = make_train_token_iterator(train_ds, tok, cfg.train.seed)
    train_iter = build_batches(train_tokens, seq_len, batch_size, device)

    val_ids = pack_val_tokens(val_ds, tok)

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    wb = None
    if use_wandb:
        import wandb

        wb = wandb.init(project="tiny-swap-bench", config=runconfig_to_dict(cfg), dir=str(run_dir))

    history: list[dict[str, Any]] = []
    cum_tokens = 0
    cum_train_flops = 0.0
    step = 0
    wall_t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    fwd_flops_micro = forward_flops_per_forward(cfg.model, batch=batch_size, seq_len=seq_len)
    train_flops_micro = training_flops_from_forward(fwd_flops_micro)

    budget_tokens = cfg.eval.flop_matched_budget_tokens
    steps_for_budget = math.ceil(float(budget_tokens) / float(tokens_per_step))
    budget_train_flops = float(steps_for_budget * accum * train_flops_micro)

    model.train()
    while cum_tokens < max_tokens:
        step_lr_mult = cosine_lr_multiplier(step, cfg.train.warmup_steps, total_steps_est)
        optimizer_step_lr(opt, cfg.train.lr, step_lr_mult)

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_tokens = make_train_token_iterator(train_ds, tok, cfg.train.seed + step)
                train_iter = build_batches(train_tokens, seq_len, batch_size, device)
                batch = next(train_iter)

            x = batch.input_ids  # (B, T)
            y = batch.labels  # (B, T)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)):
                logits = model(x)  # (B, T, V)
                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)
                loss = F.cross_entropy(logits_flat, y_flat)

            (loss / accum).backward()
            loss_accum += float(loss.detach().item())

            cum_tokens += x.numel()
            cum_train_flops += float(train_flops_micro)

            if cum_tokens >= max_tokens:
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

        wall_seconds = time.perf_counter() - wall_t0

        should_eval = (step > 0 and step % cfg.eval.eval_interval_steps == 0) or (smoke and step <= 2)
        should_ckpt = step > 0 and step % cfg.eval.checkpoint_interval_steps == 0

        if should_eval:
            model.eval()
            amp_enabled = dtype == torch.bfloat16
            vbatches = build_batches(iter(val_ids), seq_len, batch_size, device)
            val_loss, used_batches = batches_loss(
                model,
                vbatches,
                device=device,
                dtype=dtype,
                max_batches=smoke_eval_batches if smoke else 100,
                amp_enabled=amp_enabled,
            )
            peak_mem = torch.cuda.max_memory_allocated(device)
            record = {
                "step": step,
                "cum_tokens": cum_tokens,
                "cum_train_flops": cum_train_flops,
                "wall_seconds": wall_seconds,
                "train_loss_batch": loss_accum / accum,
                "val_loss": val_loss,
                "val_batches_used": used_batches,
                "tokens_per_sec": cum_tokens / max(wall_seconds, 1e-9),
                "peak_gpu_memory_bytes": int(peak_mem),
            }
            history.append(record)
            if wb is not None:
                wb.log(record)
            model.train()

        if should_ckpt:
            ckpt = {
                "model": model.state_dict(),
                "cfg": runconfig_to_dict(cfg),
                "cum_tokens": cum_tokens,
                "cum_train_flops": cum_train_flops,
                "wall_seconds": wall_seconds,
                "step": step,
            }
            if isinstance(opt, MuonHybrid):
                ckpt["optimizer_muon"] = opt.state_dict()
            else:
                ckpt["optimizer"] = opt.state_dict()
            torch.save(ckpt, run_dir / f"checkpoint_step_{step}.pt")

        step += 1
        if cum_tokens >= max_tokens:
            break

    wall_total = time.perf_counter() - wall_t0

    model.eval()
    vbatches_final = build_batches(iter(val_ids), seq_len, batch_size, device)
    final_val_loss, _ = batches_loss(
        model,
        vbatches_final,
        device=device,
        dtype=dtype,
        max_batches=smoke_eval_batches if smoke else 200,
        amp_enabled=dtype == torch.bfloat16,
    )

    flop_matched_val_loss = select_matched_metric(history, key_cum="cum_train_flops", target=budget_train_flops)

    tokens_per_sec_observed = cum_tokens / max(wall_total, 1e-9)
    wall_seconds_for_budget_tokens = float(budget_tokens) / max(tokens_per_sec_observed, 1e-9)
    wall_matched_val_loss = select_matched_metric(history, key_cum="wall_seconds", target=wall_seconds_for_budget_tokens)

    prompts_file = prompts_path if prompts_path is not None else Path(__file__).resolve().parents[3] / "eval" / "prompts.json"
    prompts_list, _rubric = load_prompts_and_rubric(prompts_file)
    enc = tok.enc
    judge_scores = []
    for prompt in prompts_list[: cfg.eval.generation_num_samples]:
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        out_ids = generate_completion(model, ids, max_new_tokens=64, device=device, dtype=dtype)
        text = enc.decode(out_ids[0].tolist())
        judge_scores.append(score_completion_with_env(text, prompt))

    judge_agg = aggregate_judge_scores(judge_scores)

    summary = {
        "seed": cfg.train.seed,
        "non_embedding_params": non_emb,
        "total_trainable_params": total_p,
        "final_val_loss": final_val_loss,
        "budget_train_tokens": budget_tokens,
        "budget_train_flops_counter": budget_train_flops,
        "val_loss_at_flop_checkpoint": flop_matched_val_loss,
        "wall_seconds_total_run": wall_total,
        "wall_seconds_for_budget_tokens_throughput_estimate": wall_seconds_for_budget_tokens,
        "val_loss_at_wall_checkpoint": wall_matched_val_loss,
        "cum_train_flops_observed": cum_train_flops,
        "tokens_per_sec_mean": tokens_per_sec_observed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "llm_judge": judge_agg,
        "history_len": len(history),
    }

    (run_dir / "results_summary.json").write_text(json.dumps(summary, indent=2), default=str)

    if wb is not None:
        wb.finish()

    return summary


def select_matched_metric(history: list[dict[str, Any]], *, key_cum: str, target: float) -> float | None:
    """Pick validation loss from record whose cumulative metric is closest without exceeding target."""
    eligible = [h for h in history if isinstance(h.get("val_loss"), (float, int)) and key_cum in h]
    if not eligible:
        return None
    below = [h for h in eligible if float(h[key_cum]) <= target]
    if not below:
        return float(eligible[0]["val_loss"])
    best = max(below, key=lambda h: float(h[key_cum]))
    return float(best["val_loss"])
