#!/usr/bin/env python3
"""Pipeline-validation baseline: reference config × seeds {0,1,2}.

Requires CUDA (training refuses CPU). Use ``--smoke`` for a tiny budget while keeping bf16 paths hot.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import uuid
from dataclasses import replace
from pathlib import Path

from tiny_swap_bench.config_schema import load_run_config
from tiny_swap_bench.training.loop import train_run


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    return statistics.stdev(values) / math.sqrt(n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-root", type=Path, default=Path(__file__).resolve().parents[2] / "configs")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_cfg = load_run_config([args.configs_root / "base.yaml"])
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    exp_root = Path(__file__).resolve().parent / run_id
    prompts_path = Path(__file__).resolve().parents[2] / "eval" / "prompts.json"

    rows: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        cfg = replace(base_cfg, train=replace(base_cfg.train, seed=seed))
        run_dir = exp_root / f"seed_{seed}"
        summary = train_run(cfg, run_dir, prompts_path=prompts_path, smoke=args.smoke, smoke_eval_batches=1)
        summary["seed"] = seed
        rows.append(summary)

    def collect_float(key: str) -> tuple[list[float], list[float | None]]:
        raw = [row.get(key) for row in rows]
        nums = [float(x) for x in raw if isinstance(x, (float, int)) and x is not None]
        optional = [float(x) if isinstance(x, (float, int)) else None for x in raw]
        return nums, optional

    fv, _ = collect_float("final_val_loss")
    ff_nums, ff_raw = collect_float("val_loss_at_flop_checkpoint")
    vw_nums, vw_raw = collect_float("val_loss_at_wall_checkpoint")
    tps, _ = collect_float("tokens_per_sec_mean")
    mem, _ = collect_float("peak_gpu_memory_bytes")

    agg = {
        "run_id": run_id,
        "final_val_loss": {"mean": statistics.mean(fv), "stderr": _stderr(fv), "per_seed": fv},
        "val_loss_at_flop_checkpoint": {
            "mean": statistics.mean(ff_nums) if ff_nums else None,
            "stderr": _stderr(ff_nums) if len(ff_nums) > 1 else float("nan"),
            "per_seed": ff_raw,
        },
        "val_loss_at_wall_checkpoint": {
            "mean": statistics.mean(vw_nums) if vw_nums else None,
            "stderr": _stderr(vw_nums) if len(vw_nums) > 1 else float("nan"),
            "per_seed": vw_raw,
        },
        "tokens_per_sec_mean": {"mean": statistics.mean(tps), "stderr": _stderr(tps), "per_seed": tps},
        "peak_gpu_memory_bytes": {"mean": statistics.mean(mem), "stderr": _stderr(mem), "per_seed": mem},
    }

    out_json = exp_root / "results_table.json"
    out_json.write_text(json.dumps({"run_id": run_id, "rows": rows, "aggregate": agg}, indent=2), default=str)

    print("\n### Baseline reproducibility — results (mean ± stderr over seeds)\n")
    print("| Quantity | seed 0 | seed 1 | seed 2 | mean ± stderr |")
    print("| --- | --- | --- | --- | --- |")
    print(
        f"| Final val CE (nats) | {fv[0]:.5f} | {fv[1]:.5f} | {fv[2]:.5f} | "
        f"{statistics.mean(fv):.5f} ± {_stderr(fv):.5f} |"
    )
    ff_disp = [f"{x:.5f}" if x is not None else "na" for x in ff_raw]
    ff_mean_str = f"{statistics.mean(ff_nums):.5f} ± {_stderr(ff_nums):.5f}" if ff_nums else "na"
    print(f"| Val CE @ FLOP-matched checkpoint | {ff_disp[0]} | {ff_disp[1]} | {ff_disp[2]} | {ff_mean_str} |")
    vw_disp = [f"{x:.5f}" if x is not None else "na" for x in vw_raw]
    vw_mean_str = f"{statistics.mean(vw_nums):.5f} ± {_stderr(vw_nums):.5f}" if vw_nums else "na"
    print(f"| Val CE @ wall-matched checkpoint | {vw_disp[0]} | {vw_disp[1]} | {vw_disp[2]} | {vw_mean_str} |")
    print(
        f"| Throughput (tok/s) | {tps[0]:.2f} | {tps[1]:.2f} | {tps[2]:.2f} | "
        f"{statistics.mean(tps):.2f} ± {_stderr(tps):.2f} |"
    )
    print(
        f"| Peak GPU memory (bytes) | {int(mem[0])} | {int(mem[1])} | {int(mem[2])} | "
        f"{int(statistics.mean(mem))} ± {_stderr(mem):.0f} |"
    )
    print(f"\nArtifacts: `{exp_root}` — `{out_json}`\n")


if __name__ == "__main__":
    main()
