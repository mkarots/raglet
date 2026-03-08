#!/usr/bin/env python3
"""Config-driven benchmark sweep runner.

Reads a YAML config that defines which benchmarks to run and what parameter
grids to sweep.  Each combination in the grid is executed as a subprocess,
and results are collected into a single JSON report.

Usage:
    uv run python benchmarks/sweep.py
    uv run python benchmarks/sweep.py --config benchmarks/sweep-quick.yaml
    uv run python benchmarks/sweep.py --only latency
    uv run python benchmarks/sweep.py --only latency,embedding-throughput
    uv run python benchmarks/sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML or JSON sweep config."""
    p = Path(path)
    if not p.exists():
        print(f"Error: config not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = p.read_text()

    if p.suffix in (".yaml", ".yml"):
        if yaml is None:
            print(
                "Error: pyyaml is required for YAML configs. "
                "Install with: uv pip install pyyaml",
                file=sys.stderr,
            )
            sys.exit(1)
        return yaml.safe_load(text)

    return json.loads(text)


def expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a parameter grid into all combinations.

    Values that are not lists are treated as single-element lists.
    """
    keys = list(grid.keys())
    values = []
    for k in keys:
        v = grid[k]
        values.append(v if isinstance(v, list) else [v])

    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def build_argv(script: str, params: dict[str, Any]) -> list[str]:
    """Turn a script path + param dict into a subprocess argv."""
    argv = [sys.executable, script]
    for key, val in params.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, list):
            argv.append(flag)
            argv.extend(str(v) for v in val)
        else:
            argv.extend([flag, str(val)])
    return argv


def run_one(
    name: str,
    argv: list[str],
    params: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one benchmark run and return a result record."""
    cmd_str = " ".join(argv)

    if dry_run:
        print(f"  [dry-run] {cmd_str}")
        return {
            "benchmark": name,
            "params": params,
            "command": cmd_str,
            "status": "dry-run",
        }

    print(f"  $ {cmd_str}", flush=True)
    t0 = time.perf_counter()

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        elapsed = time.perf_counter() - t0
        ok = proc.returncode == 0

        if not ok:
            print(f"    FAILED (exit {proc.returncode}, {elapsed:.1f}s)", flush=True)
            stderr_tail = proc.stderr.strip().splitlines()[-5:] if proc.stderr else []
            for line in stderr_tail:
                print(f"    stderr: {line}", flush=True)
        else:
            print(f"    OK ({elapsed:.1f}s)", flush=True)

        return {
            "benchmark": name,
            "params": params,
            "command": cmd_str,
            "status": "pass" if ok else "fail",
            "exit_code": proc.returncode,
            "elapsed_s": round(elapsed, 2),
            "stdout_tail": proc.stdout.strip().splitlines()[-20:] if proc.stdout else [],
            "stderr_tail": proc.stderr.strip().splitlines()[-10:] if proc.stderr else [],
        }

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"    TIMEOUT ({elapsed:.1f}s)", flush=True)
        return {
            "benchmark": name,
            "params": params,
            "command": cmd_str,
            "status": "timeout",
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"    ERROR: {e}", flush=True)
        return {
            "benchmark": name,
            "params": params,
            "command": cmd_str,
            "status": "error",
            "error": str(e),
            "elapsed_s": round(elapsed, 2),
        }


def run_sweep(
    config: dict[str, Any],
    only: set[str] | None = None,
    dry_run: bool = False,
    output: str = "benchmarks/sweep-results.json",
) -> list[dict[str, Any]]:
    """Execute the full sweep from a loaded config."""
    benchmarks = config.get("benchmarks", {})
    if not benchmarks:
        print("Error: no benchmarks defined in config", file=sys.stderr)
        sys.exit(1)

    results: list[dict[str, Any]] = []
    total_runs = 0

    selected = {k: v for k, v in benchmarks.items() if only is None or k in only}
    if only and not selected:
        available = ", ".join(benchmarks.keys())
        print(f"Error: none of {only} found in config. Available: {available}", file=sys.stderr)
        sys.exit(1)

    for name, bench_cfg in selected.items():
        script = bench_cfg.get("script")
        if not script:
            print(f"  Skipping {name}: no 'script' defined", flush=True)
            continue

        if not Path(script).exists():
            print(f"  Skipping {name}: script not found: {script}", flush=True)
            continue

        grid = bench_cfg.get("grid", {})
        combos = expand_grid(grid) if grid else [{}]
        total_runs += len(combos)

    print(f"Sweep: {len(selected)} benchmarks, {total_runs} total runs")
    if dry_run:
        print("(dry run — no benchmarks will actually execute)\n")
    print()

    t_start = time.perf_counter()
    run_idx = 0

    for name, bench_cfg in selected.items():
        script = bench_cfg.get("script")
        if not script or not Path(script).exists():
            continue

        grid = bench_cfg.get("grid", {})
        combos = expand_grid(grid) if grid else [{}]

        print(f"{'─' * 60}")
        print(f"  {name}  ({len(combos)} run{'s' if len(combos) != 1 else ''})")
        print(f"{'─' * 60}")

        for params in combos:
            run_idx += 1
            print(f"\n  [{run_idx}/{total_runs}] {name}", flush=True)
            if params:
                print(f"    params: {params}", flush=True)

            argv = build_argv(script, params)
            result = run_one(name, argv, params, dry_run=dry_run)
            results.append(result)

    total_elapsed = time.perf_counter() - t_start

    # Summary
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    timed_out = sum(1 for r in results if r["status"] == "timeout")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'═' * 60}")
    print(f"  Sweep complete: {passed} passed, {failed} failed, "
          f"{timed_out} timed out, {errors} errors  ({total_elapsed:.1f}s)")
    print(f"{'═' * 60}")

    if not dry_run:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_file": str(Path(output).stem),
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "timed_out": timed_out,
                "errors": errors,
                "elapsed_s": round(total_elapsed, 2),
            },
            "runs": results,
        }
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Config-driven benchmark sweep runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python benchmarks/sweep.py
  uv run python benchmarks/sweep.py --config benchmarks/sweep-quick.yaml
  uv run python benchmarks/sweep.py --only latency
  uv run python benchmarks/sweep.py --only latency,embedding-throughput
  uv run python benchmarks/sweep.py --dry-run
  uv run python benchmarks/sweep.py --output results/sweep-2026-03-08.json
        """,
    )
    parser.add_argument(
        "--config",
        default="benchmarks/sweep.yaml",
        help="Path to sweep config (YAML or JSON, default: benchmarks/sweep.yaml)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of benchmark names to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/sweep-results.json",
        help="Path for results JSON (default: benchmarks/sweep-results.json)",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    only = set(args.only.split(",")) if args.only else None

    results = run_sweep(config, only=only, dry_run=args.dry_run, output=args.output)

    any_failed = any(r["status"] not in ("pass", "dry-run") for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
