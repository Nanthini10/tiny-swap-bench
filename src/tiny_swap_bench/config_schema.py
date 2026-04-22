"""YAML + dataclass run configuration (no Hydra)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


PeName = Literal["learned_abs", "rope", "nope", "alibi"]
NormName = Literal["pre_ln_ln", "pre_ln_rms", "post_ln_ln", "pre_ln_rms_qk"]
OptimName = Literal["adamw", "lion", "muon"]


@dataclass
class ModelConfig:
    n_layer: int = 11
    n_embd: int = 384
    n_head: int = 6
    mlp_ratio: float = 4.0
    seq_len: int = 512
    vocab_size: int = 50257
    dropout: float = 0.0
    pe: PeName = "learned_abs"
    norm: NormName = "pre_ln_ln"


@dataclass
class TrainConfig:
    lr: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_train_tokens: int = 1_000_000_000
    precision: Literal["bf16"] = "bf16"
    optimizer: OptimName = "adamw"
    seed: int = 0


@dataclass
class DataConfig:
    dataset_path: str = "roneneldan/TinyStories"
    dataset_split_train: str = "train"
    val_fraction: float = 0.01
    tokenizer_name: str = "gpt2"


@dataclass
class EvalConfig:
    eval_interval_steps: int = 500
    checkpoint_interval_steps: int = 500
    generation_num_samples: int = 100
    flop_matched_budget_tokens: int = 1_000_000_000


@dataclass
class RunConfig:
    experiment_name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dict_to_runconfig(d: dict[str, Any]) -> RunConfig:
    m = d.get("model", {}) or {}
    t = d.get("train", {}) or {}
    da = d.get("data", {}) or {}
    e = d.get("eval", {}) or {}
    return RunConfig(
        experiment_name=d.get("experiment_name", "run"),
        model=ModelConfig(**{**asdict(ModelConfig()), **m}),
        train=TrainConfig(**{**asdict(TrainConfig()), **t}),
        data=DataConfig(**{**asdict(DataConfig()), **da}),
        eval=EvalConfig(**{**asdict(EvalConfig()), **e}),
    )


def load_run_config(paths: list[Path | str]) -> RunConfig:
    if not paths:
        raise ValueError("load_run_config requires at least one YAML path")
    merged: dict[str, Any] = {}
    for path in paths:
        merged = _merge_dict(merged, load_yaml(path))
    return dict_to_runconfig(merged)


def runconfig_to_dict(cfg: RunConfig) -> dict[str, Any]:
    return {
        "experiment_name": cfg.experiment_name,
        "model": asdict(cfg.model),
        "train": asdict(cfg.train),
        "data": asdict(cfg.data),
        "eval": asdict(cfg.eval),
    }
