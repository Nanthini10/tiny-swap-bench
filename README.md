# tiny-swap-bench

Controlled small-scale ablations of decoder-transformer design choices on **TinyStories**, Karpathy-style (plain PyTorch, one training loop, YAML + dataclasses).

## Reference configuration (null hypothesis)

| Quantity | Value |
|----------|--------|
| Layers | 11 |
| Width | 384 |
| Heads | 6 (head dim 64) |
| MLP | 4× GELU |
| Sequence length | 512 |
| Tokenizer | GPT-2 BPE (`tiktoken`, vocab 50257) |
| Positional (reference) | learned absolute |
| Norm (reference) | pre-LayerNorm + LayerNorm |
| Optimizer (reference) | AdamW |
| Train tokens / run | default `1_000_000_000` |
| Batch | 64 × 512 = **32,768 tokens/step** |
| Precision | **bf16** mixed (no fp16/fp8) |
| Seeds | **{0, 1, 2}** for every experiment |

**Non-embedding parameter count (reference, tied head, excludes `wte` and learned `wpe`):** **19,481,856** (see `DESIGN.md`).

## Setup

Requires **Python ≥ 3.9** (see `pyproject.toml`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Training **requires CUDA**; the loop raises if `torch.cuda.is_available()` is false (no silent CPU fallback).

Optional: `wandb` logs only if `WANDB_API_KEY` is set.

## Layout

- `src/tiny_swap_bench/` — model (`model/`), optimizers (`optim/`), data (`data/`), eval (`eval/`), training loop (`training/loop.py`), narrow utils (`utils/`).
- `configs/` — `base.yaml` plus overrides under `pe/`, `norm/`, `optim/`.
- `experiments/` — runnable scripts; each run writes `resolved_run.json` + metrics under `experiments/<name>/<run_id>/`.
- `eval/prompts.json` — 100 fixed prompts + rubric text for LLM-judge scaffolding (API call not wired in v0).

## Tests

CPU-focused suite (deterministic cuDNN enabled only under pytest via `tests/conftest.py`):

```bash
pytest
```

CUDA-only checks are skipped when no GPU is present.

## Baseline reproducibility (pipeline validation)

Runs the reference config three times (seeds 0–1–2), writes per-seed dirs and `results_table.json`.

```bash
python experiments/baseline_reproducibility/run.py --smoke   # tiny token budget (~8k tokens/run)
python experiments/baseline_reproducibility/run.py           # full 1B-token budget per seed (~hours on one H100)
```

`--smoke` caps tokens per seed for wiring checks; full runs use `configs/base.yaml` (`max_train_tokens: 1000000000`).

## Swap axes (v1)

- **PE:** `learned_abs`, `rope`, `nope`, `alibi` (`configs/pe/*.yaml`).
- **Norm:** `pre_ln_ln`, `pre_ln_rms`, `post_ln_ln`, `pre_ln_rms_qk` (`configs/norm/*.yaml`).
- **Optimizer:** `adamw`, `lion`, `muon` (`configs/optim/*.yaml`).

Merge any override with base:

```python
from pathlib import Path
from tiny_swap_bench.config_schema import load_run_config
cfg = load_run_config([Path("configs/base.yaml"), Path("configs/pe/rope.yaml")])
```

## Metrics (per run)

Every run records (per seed in aggregate scripts): final validation CE, validation CE at FLOP-matched checkpoint (training FLOPs = 3× forward per micro-batch, cumulative), validation CE at wall-matched checkpoint (seconds estimated from throughput to reach budget tokens), LLM-judge stub scores, analytical FLOPs counter, wall time, tokens/sec, peak GPU memory. See `results_summary.json` under each run directory.

## License

Research prototype — add a license before public release.
