"""Write resolved config, environment, and git SHA to a run directory."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tiny_swap_bench.config_schema import RunConfig, runconfig_to_dict


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path.cwd())
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def write_run_metadata(out_dir: Path, cfg: RunConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "config": runconfig_to_dict(cfg),
        "environment": environment_info(),
        "git_sha": _git_sha(),
    }
    (out_dir / "resolved_run.json").write_text(json.dumps(resolved, indent=2, default=str), encoding="utf-8")
