"""Tiny end-to-end training smoke on CUDA with monkeypatched data (no HuggingFace IO)."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from tiny_swap_bench.config_schema import DataConfig, EvalConfig, ModelConfig, RunConfig, TrainConfig
from tiny_swap_bench.training import loop as train_loop


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for train_run")
def test_train_run_smoke_monkeypatch(tmp_path):
    texts = [{"text": ("once upon a time " * 120)} for _ in range(48)]
    train_ds = Dataset.from_list(texts[:40])
    val_ds = Dataset.from_list(texts[40:])

    def _fake_load_train_val_rows(*_a, **_k):
        return train_ds, val_ds

    cfg = RunConfig(
        experiment_name="e2e_smoke",
        model=ModelConfig(n_layer=2, n_embd=64, n_head=4, seq_len=64, vocab_size=50257, pe="learned_abs", norm="pre_ln_ln"),
        train=TrainConfig(
            batch_size=2,
            grad_accum_steps=1,
            max_train_tokens=8192,
            warmup_steps=10,
            optimizer="adamw",
            seed=0,
        ),
        data=DataConfig(dataset_path="stub/stub"),
        eval=EvalConfig(eval_interval_steps=1, checkpoint_interval_steps=9999, generation_num_samples=3),
    )
    prompts = Path(__file__).resolve().parents[1] / "eval" / "prompts.json"

    with patch.object(train_loop, "load_train_val_rows", _fake_load_train_val_rows):
        out = train_loop.train_run(cfg, tmp_path, prompts_path=prompts, smoke=True, smoke_eval_batches=1)

    assert out["final_val_loss"] == out["final_val_loss"]
    assert (tmp_path / "resolved_run.json").is_file()
