"""Every YAML under ``configs/`` merges with defaults without error."""

from pathlib import Path

import pytest

from tiny_swap_bench.config_schema import load_run_config


def _all_yaml_files(root: Path):
    return sorted(root.rglob("*.yaml"))


def test_all_config_yamls_resolve():
    root = Path(__file__).resolve().parents[1] / "configs"
    files = _all_yaml_files(root)
    assert files, "expected configs/**/*.yaml"
    base = root / "base.yaml"
    for path in files:
        if path.name == "base.yaml":
            load_run_config([path])
        else:
            load_run_config([base, path])
