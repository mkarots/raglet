#!/usr/bin/env python3
"""Generate a release-ready markdown benchmark report.

Reads JSON results from the embedding-throughput and latency benchmarks,
optionally compares against a baseline, and produces a markdown summary
suitable for GitHub release notes or documentation.

Usage:
    uv run python benchmarks/report.py
    uv run python benchmarks/report.py --baseline benchmarks/baselines/v0.2.0.json
    uv run python benchmarks/report.py --output BENCHMARK_REPORT.md
    uv run python benchmarks/report.py --format github  # wraps in <details> tag
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_RESULTS = "benchmarks/embedding-throughput/results.json"
DEFAULT_LATENCY_RESULTS = "benchmarks/latency/results.json"


def _load_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _fmt_bytes(n: int) -> str:
    if n >= 1_048_576:
        return f"{n / 1_048_576:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def _fmt_time(seconds: float) -> str:
    if seconds < 0.01:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 1.0:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _fmt_ms(ms: float) -> str:
    if ms < 1.0:
        return f"{ms:.2f} ms"
    if ms < 100:
        return f"{ms:.1f} ms"
    return f"{ms:.0f} ms"


def _delta_str(current: float, baseline: float) -> str:
    """Format a percentage change string like '+12%' or '-5%'."""
    if baseline == 0:
        return "—"
    pct = ((current - baseline) / baseline) * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.0f}%"


def _match_by_bytes(rows: list[dict], target_bytes: int) -> dict | None:
    """Find the row whose text_bytes is closest to target_bytes."""
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["text_bytes"] - target_bytes))


def _build_config_section(embed_data: dict | None) -> str:
    lines: list[str] = []
    if embed_data and "config" in embed_data:
        cfg = embed_data["config"]
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        lines.append(f"| Model | `{cfg.get('model', '?')}` |")
        lines.append(f"| Device | {cfg.get('device', '?')} |")
        lines.append(f"| Batch size | {cfg.get('batch_size', '?')} |")
        lines.append(f"| FP16 | {cfg.get('fp16', '?')} |")
        lines.append(
            f"| Chunk size | {cfg.get('chunk_size', '?')} tokens "
            f"(~{cfg.get('chunk_size', 0) * 4} chars) |"
        )
        lines.append(f"| Chunk overlap | {cfg.get('chunk_overlap', '?')} tokens |")
        lines.append(f"| Chunk strategy | {cfg.get('chunk_strategy', '?')} |")
        lines.append(f"| Max sequence length | {cfg.get('max_seq_length', '?')} |")
    return "\n".join(lines)


def _build_throughput_table(
    embed_rows: list[dict],
    baseline_rows: list[dict] | None = None,
) -> str:
    has_baseline = baseline_rows is not None and len(baseline_rows) > 0

    header = "| Text size | Chunks | Embed time | Throughput (chunks/s) |"
    sep = "|---------:|-------:|-----------:|---------------------:|"
    if has_baseline:
        header += " vs baseline |"
        sep += "------------:|"

    lines = [header, sep]

    for row in embed_rows:
        size = row["text_size"]
        chunks = f"{row['chunks']:,}"
        embed_t = _fmt_time(row["mean_embed_s"])
        tp = f"{row['mean_chunks_per_s']:,.0f}"

        line = f"| {size} | {chunks} | {embed_t} | {tp} |"

        if has_baseline:
            match = _match_by_bytes(baseline_rows, row["text_bytes"])
            if match:
                delta = _delta_str(row["mean_chunks_per_s"], match["mean_chunks_per_s"])
                line += f" {delta} |"
            else:
                line += " — |"

        lines.append(line)

    return "\n".join(lines)


def _build_latency_table(
    latency_rows: list[dict],
    baseline_rows: list[dict] | None = None,
) -> str:
    has_baseline = baseline_rows is not None and len(baseline_rows) > 0

    header = (
        "| Text size | Chunks | Build | Search p50 | Save p50 | Load p50 |"
    )
    sep = "|---------:|-------:|------:|-----------:|---------:|---------:|"
    if has_baseline:
        header += " Search delta |"
        sep += "-------------:|"

    lines = [header, sep]

    for row in latency_rows:
        size = row["text_size"]
        chunks = f"{row['chunks']:,}"
        build = _fmt_time(row["build_s"])
        search_p50 = _fmt_ms(row["search"]["p50_ms"])
        save_p50 = _fmt_ms(row["save"]["p50_ms"])
        load_p50 = _fmt_ms(row["load"]["p50_ms"])

        line = f"| {size} | {chunks} | {build} | {search_p50} | {save_p50} | {load_p50} |"

        if has_baseline:
            match = _match_by_bytes(baseline_rows, row["text_bytes"])
            if match:
                delta = _delta_str(row["search"]["p50_ms"], match["search"]["p50_ms"])
                line += f" {delta} |"
            else:
                line += " — |"

        lines.append(line)

    return "\n".join(lines)


def _build_highlights(
    embed_rows: list[dict],
    latency_rows: list[dict],
) -> str:
    lines: list[str] = []

    if embed_rows:
        peak_tp = max(r["mean_chunks_per_s"] for r in embed_rows)
        largest = embed_rows[-1]
        lines.append(
            f"- Peak embedding throughput: **{peak_tp:,.0f} chunks/s** "
            f"(at {largest['text_size']})"
        )

    if latency_rows:
        all_search = [r["search"]["p50_ms"] for r in latency_rows]
        min_search = min(all_search)
        max_search = max(all_search)
        lines.append(
            f"- Search p50: **{_fmt_ms(min_search)}** – **{_fmt_ms(max_search)}** "
            f"across all sizes"
        )

        largest_lat = latency_rows[-1]
        lines.append(
            f"- At {largest_lat['text_size']} ({largest_lat['chunks']:,} chunks): "
            f"build {_fmt_time(largest_lat['build_s'])}, "
            f"search {_fmt_ms(largest_lat['search']['p50_ms'])}, "
            f"save {_fmt_ms(largest_lat['save']['p50_ms'])}, "
            f"load {_fmt_ms(largest_lat['load']['p50_ms'])}"
        )

    if latency_rows:
        lines.append(
            "- Building is the expensive one-time cost; "
            "search/save/load stay fast regardless of size"
        )

    return "\n".join(lines)


def generate_report(
    embed_path: str = DEFAULT_EMBEDDING_RESULTS,
    latency_path: str = DEFAULT_LATENCY_RESULTS,
    baseline_path: str | None = None,
    fmt: str = "plain",
) -> str:
    embed_data = _load_json(embed_path)
    latency_data = _load_json(latency_path)

    if embed_data is None and latency_data is None:
        return (
            "No benchmark results found. Run benchmarks first:\n\n"
            "    make benchmark          # embedding throughput\n"
            "    uv run python benchmarks/latency/run.py   # latency\n"
        )

    embed_rows = embed_data.get("results", []) if isinstance(embed_data, dict) else []
    if isinstance(latency_data, dict):
        latency_rows = latency_data.get("results", [])
    elif isinstance(latency_data, list):
        latency_rows = latency_data
    else:
        latency_rows = []

    baseline = _load_json(baseline_path) if baseline_path else None
    baseline_embed: list[dict] | None = None
    baseline_latency: list[dict] | None = None
    if baseline:
        baseline_embed = baseline.get("embedding", []) or None
        baseline_latency = baseline.get("latency", []) or None

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sections: list[str] = []

    sections.append(f"## Benchmark Results ({ts})\n")

    # Config
    if embed_rows:
        sections.append("### Configuration\n")
        sections.append(_build_config_section(embed_data))
        sections.append("")

    # Highlights
    if embed_rows or latency_rows:
        sections.append("### Highlights\n")
        sections.append(_build_highlights(embed_rows, latency_rows))
        sections.append("")

    # Throughput
    if embed_rows:
        sections.append("### Embedding Throughput\n")
        sections.append(_build_throughput_table(embed_rows, baseline_embed))
        sections.append("")

    # Latency
    if latency_rows:
        sections.append("### Operation Latency\n")
        sections.append(_build_latency_table(latency_rows, baseline_latency))
        sections.append("")

    # Interpretation
    sections.append("### How to read these numbers\n")
    sections.append(
        "- **Build** is a one-time cost when creating a `.raglet` file from scratch.\n"
        "  Appending new content (`add_file` / `add_text`) only embeds the new chunks.\n"
        "- **Search** stays under 10 ms at workspace scale — "
        "it does not grow with dataset size (FAISS indexed).\n"
        "- **Save/Load** scale linearly with chunk count but stay well under 100 ms "
        "for typical workspaces."
    )

    report = "\n".join(sections)

    if fmt == "github":
        report = (
            "<details>\n<summary>Benchmark results</summary>\n\n"
            + report
            + "\n</details>"
        )

    return report


def save_baseline(
    embed_path: str,
    latency_path: str,
    output_path: str,
) -> None:
    """Snapshot current results as a baseline for future comparisons."""
    embed_data = _load_json(embed_path)
    latency_data = _load_json(latency_path)

    baseline: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(embed_data, dict):
        baseline["config"] = embed_data.get("config", {})
        baseline["embedding"] = embed_data.get("results", [])
    if isinstance(latency_data, dict):
        baseline["latency"] = latency_data.get("results", [])
    elif isinstance(latency_data, list):
        baseline["latency"] = latency_data

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline saved to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate release-ready benchmark report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  uv run python benchmarks/report.py
  uv run python benchmarks/report.py --output BENCHMARK_REPORT.md
  uv run python benchmarks/report.py --baseline benchmarks/baselines/v0.2.0.json
  uv run python benchmarks/report.py --format github
  uv run python benchmarks/report.py --save-baseline benchmarks/baselines/v0.3.0.json
""",
    )
    parser.add_argument(
        "--embedding",
        default=DEFAULT_EMBEDDING_RESULTS,
        help=f"Path to embedding-throughput results JSON (default: {DEFAULT_EMBEDDING_RESULTS})",
    )
    parser.add_argument(
        "--latency",
        default=DEFAULT_LATENCY_RESULTS,
        help=f"Path to latency results JSON (default: {DEFAULT_LATENCY_RESULTS})",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline JSON for delta comparison",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "github"],
        default="plain",
        help="Output format: plain markdown or github (wrapped in <details>)",
    )
    parser.add_argument(
        "--save-baseline",
        default=None,
        metavar="PATH",
        help="Snapshot current results as a baseline JSON file, then exit",
    )

    args = parser.parse_args()

    if args.save_baseline:
        save_baseline(args.embedding, args.latency, args.save_baseline)
        return 0

    report = generate_report(
        embed_path=args.embedding,
        latency_path=args.latency,
        baseline_path=args.baseline,
        fmt=args.format,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n")
        print(f"Report written to {out}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
