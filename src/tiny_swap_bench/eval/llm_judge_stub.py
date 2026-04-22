"""LLM-as-judge scoring (stub unless API credentials are provided)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RubricScores:
    coherence: float  # [0, 10]
    grammar: float  # [0, 10]
    creativity: float  # [0, 10]


def load_prompts_and_rubric(prompts_path: Path) -> tuple[list[str], dict[str, str]]:
    payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    rubric = payload["rubric"]
    if len(prompts) != 100:
        raise ValueError(f"Expected 100 prompts in {prompts_path}, got {len(prompts)}.")
    return prompts, rubric


def score_completion_stub(completion_text: str) -> RubricScores:
    """Deterministic placeholder when no API key is configured."""
    _ = completion_text
    return RubricScores(coherence=0.0, grammar=0.0, creativity=0.0)


def score_completion_with_env(completion_text: str, prompt: str) -> RubricScores:
    """If ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY`` is set, call provider; else stub."""
    _ = prompt
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        raise NotImplementedError(
            "LLM judge API calls are intentionally not implemented in v0; "
            "set keys only after wiring a provider implementation."
        )
    return score_completion_stub(completion_text)


def aggregate_judge_scores(scores: list[RubricScores]) -> dict[str, float]:
    if not scores:
        return {"coherence_mean": float("nan"), "grammar_mean": float("nan"), "creativity_mean": float("nan")}
    n = len(scores)
    return {
        "coherence_mean": sum(s.coherence for s in scores) / n,
        "grammar_mean": sum(s.grammar for s in scores) / n,
        "creativity_mean": sum(s.creativity for s in scores) / n,
    }
