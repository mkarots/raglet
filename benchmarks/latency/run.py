"""Latency benchmark: p50 / p95 / p99 for save, load, and search.

Measures raglet operations across text sizes from 5 KB to 5 MB, with each
size annotated by the approximate LLM token equivalent.  This demonstrates
that raglet retrieval gives sub-second access to knowledge that would
otherwise require stuffing hundreds of thousands of tokens into a context
window.

Usage:
    uv run python benchmarks/latency/run.py
    uv run python benchmarks/latency/run.py --runs 30 --format sqlite
    uv run python benchmarks/latency/run.py --chunk-size 128 --model all-mpnet-base-v2
    uv run python benchmarks/latency/run.py --sizes 0.1 1.0 5.0
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from raglet import RAGlet
from raglet.config.config import RAGletConfig

CHARS_PER_LLM_TOKEN = 4

DEFAULT_SIZES_MB = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


def _sizes_to_spec(sizes_mb: list[float]) -> list[dict[str, Any]]:
    """Convert a list of MB values to the internal size-spec format."""
    specs: list[dict[str, Any]] = []
    for mb in sorted(sizes_mb):
        target = int(mb * 1024 * 1024)
        if target >= 1_048_576:
            label = f"{target / 1_048_576:.0f} MB"
        else:
            label = f"{target / 1024:.0f} KB"
        specs.append({"label": label, "bytes": target})
    return specs

SEARCH_QUERIES = [
    "database connection pooling",
    "error handling best practices",
    "performance optimisation techniques",
    "user authentication flow",
    "deployment configuration",
]

PARAGRAPH = (
    "The system processes incoming requests through a pipeline of middleware "
    "components. Each component validates headers, authenticates the caller, "
    "checks rate limits, and routes to the appropriate handler. Handlers "
    "interact with the data layer through repository interfaces, keeping "
    "business logic decoupled from storage concerns. Configuration is loaded "
    "from environment variables at startup and validated against a schema. "
    "Errors propagate as typed exceptions with machine-readable codes. "
    "The deployment pipeline runs linting, type checking, and integration "
    "tests before publishing artifacts to the registry.\n\n"
)


def generate_text_files(target_bytes: int, out_dir: Path) -> list[str]:
    """Create realistic text files totalling approximately *target_bytes*."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    written = 0
    paths: list[str] = []
    file_idx = 0
    para_bytes = len(PARAGRAPH.encode())

    while written < target_bytes:
        fp = out_dir / f"doc_{file_idx:04d}.txt"
        paths.append(str(fp))
        with open(fp, "w") as f:
            file_written = 0
            while written < target_bytes and file_written < 50_000:
                f.write(PARAGRAPH)
                written += para_bytes
                file_written += para_bytes
        file_idx += 1

    return paths


def percentile(data: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *data* (0–100 scale)."""
    n = len(data)
    if n == 0:
        return 0.0
    s = sorted(data)
    k = (pct / 100) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def run_benchmark(
    runs: int = 20,
    storage_format: str = "sqlite",
    output_json: str = "benchmarks/latency/results.json",
    sizes_mb: list[float] | None = None,
    chunk_size: int | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Run the full latency benchmark and return results.

    Args:
        runs: Number of iterations per size point.
        storage_format: Storage backend ("sqlite" or "directory").
        output_json: Path to save results JSON.
        sizes_mb: Text sizes to test in MB. Defaults to DEFAULT_SIZES_MB.
        chunk_size: Override chunk size in tokens. Uses config default if None.
        model: Override embedding model name. Uses config default if None.
    """
    config = RAGletConfig()
    if chunk_size is not None:
        config.chunking.size = chunk_size
    if model is not None:
        config.embedding.model = model

    text_sizes = _sizes_to_spec(sizes_mb or DEFAULT_SIZES_MB)

    work_dir = Path("benchmarks/latency/_bench_tmp")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    ext = ".sqlite" if storage_format == "sqlite" else ""
    all_results: list[dict[str, Any]] = []

    cfg_chunk_size = config.chunking.size
    cfg_overlap = config.chunking.overlap
    cfg_strategy = config.chunking.strategy
    cfg_model = config.embedding.model
    cfg_device = config.embedding.device
    cfg_fp16 = config.embedding.use_fp16
    cfg_batch_size = config.embedding.batch_size

    print("=" * 100)
    print(f"Latency benchmark — {runs} runs per size, format={storage_format}")
    print(f"  model: {cfg_model}  device: {cfg_device}  fp16: {cfg_fp16}  batch_size: {cfg_batch_size}")
    print(f"  chunk_size: {cfg_chunk_size} tokens (~{cfg_chunk_size * 4} chars), "
          f"overlap: {cfg_overlap}, strategy: {cfg_strategy}")
    print("=" * 100)

    raglet_instance: RAGlet | None = None

    try:
        for size_info in text_sizes:
            label = size_info["label"]
            target = size_info["bytes"]
            approx_tokens = target // CHARS_PER_LLM_TOKEN

            print(f"\n{'─' * 100}")
            print(f"  {label}  (~{approx_tokens:,} LLM tokens)")
            print(f"{'─' * 100}")

            files_dir = work_dir / f"files_{label.replace(' ', '_')}"
            file_paths = generate_text_files(target, files_dir)

            save_path = str(work_dir / f"bench_{label.replace(' ', '_')}{ext}")

            # --- Build once (not timed in the latency loop) ---
            print(f"  Building raglet from {len(file_paths)} files …", end=" ", flush=True)
            t0 = time.perf_counter()
            raglet_instance = RAGlet.from_files(file_paths, config=config)
            build_s = time.perf_counter() - t0
            n_chunks = len(raglet_instance.chunks)
            print(f"{n_chunks} chunks in {build_s:.2f}s")

            # Initial save so load has something to read
            raglet_instance.save(save_path)

            save_times: list[float] = []
            load_times: list[float] = []
            search_times: list[float] = []

            print(f"  Running {runs} iterations …", end=" ", flush=True)
            for i in range(runs):
                # --- save ---
                sp = str(work_dir / f"bench_run_{i}{ext}")
                t0 = time.perf_counter()
                raglet_instance.save(sp)
                save_times.append(time.perf_counter() - t0)

                # --- load ---
                t0 = time.perf_counter()
                loaded = RAGlet.load(sp)
                load_times.append(time.perf_counter() - t0)
                loaded.close()
                del loaded

                # --- search (cycle through queries) ---
                q = SEARCH_QUERIES[i % len(SEARCH_QUERIES)]
                t0 = time.perf_counter()
                raglet_instance.search(q, top_k=5)
                search_times.append(time.perf_counter() - t0)

                # Clean up save artefact
                p = Path(sp)
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)

            print("done")

            # Clean up the initial save artefact
            p = Path(save_path)
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

            def fmt_pcts(data: list[float]) -> dict[str, float]:
                return {
                    "p50_ms": round(percentile(data, 50) * 1000, 2),
                    "p95_ms": round(percentile(data, 95) * 1000, 2),
                    "p99_ms": round(percentile(data, 99) * 1000, 2),
                    "mean_ms": round(statistics.mean(data) * 1000, 2),
                }

            row = {
                "text_size": label,
                "text_bytes": target,
                "approx_llm_tokens": approx_tokens,
                "chunks": n_chunks,
                "build_s": round(build_s, 2),
                "save": fmt_pcts(save_times),
                "load": fmt_pcts(load_times),
                "search": fmt_pcts(search_times),
            }
            all_results.append(row)

            # Print row
            s = row["save"]
            l = row["load"]
            q = row["search"]
            print(
                f"  {'':>10} {'p50':>10} {'p95':>10} {'p99':>10} {'mean':>10}"
            )
            print(
                f"  {'save':>10} {s['p50_ms']:>9.2f}ms {s['p95_ms']:>9.2f}ms "
                f"{s['p99_ms']:>9.2f}ms {s['mean_ms']:>9.2f}ms"
            )
            print(
                f"  {'load':>10} {l['p50_ms']:>9.2f}ms {l['p95_ms']:>9.2f}ms "
                f"{l['p99_ms']:>9.2f}ms {l['mean_ms']:>9.2f}ms"
            )
            print(
                f"  {'search':>10} {q['p50_ms']:>9.2f}ms {q['p95_ms']:>9.2f}ms "
                f"{q['p99_ms']:>9.2f}ms {q['mean_ms']:>9.2f}ms"
            )

            # Free resources before next size
            raglet_instance.close()
            raglet_instance = None
            gc.collect()

    finally:
        if raglet_instance is not None:
            raglet_instance.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    # ── Summary table ──
    print("\n" + "=" * 100)
    print("SUMMARY — all latencies in milliseconds")
    print("=" * 100)

    hdr = (
        f"{'Size':>8} {'~LLM tokens':>14} {'Chunks':>8} │"
        f" {'save p50':>9} {'p95':>9} {'p99':>9} │"
        f" {'load p50':>9} {'p95':>9} {'p99':>9} │"
        f" {'search p50':>10} {'p95':>9} {'p99':>9}"
    )
    print(hdr)
    print("─" * len(hdr))

    for r in all_results:
        s, l, q = r["save"], r["load"], r["search"]
        tok_str = f"{r['approx_llm_tokens']:,}"
        print(
            f"{r['text_size']:>8} {tok_str:>14} {r['chunks']:>8,} │"
            f" {s['p50_ms']:>9.2f} {s['p95_ms']:>9.2f} {s['p99_ms']:>9.2f} │"
            f" {l['p50_ms']:>9.2f} {l['p95_ms']:>9.2f} {l['p99_ms']:>9.2f} │"
            f" {q['p50_ms']:>10.2f} {q['p95_ms']:>9.2f} {q['p99_ms']:>9.2f}"
        )

    print(
        f"\nModel: {cfg_model} on {cfg_device} "
        f"(batch_size={cfg_batch_size}, fp16={cfg_fp16})"
    )
    print(
        f"Chunks: {cfg_chunk_size} tokens (~{cfg_chunk_size * 4} chars), "
        f"overlap {cfg_overlap} tokens, strategy {cfg_strategy}"
    )
    print(
        "Note: 'LLM tokens' approximated at ~4 chars/token (GPT-family BPE)."
    )
    print(
        "raglet search returns top-k chunks in milliseconds — the same "
        "content would consume the full context window of most LLMs."
    )

    # Save JSON
    output_data = {
        "config": {
            "model": cfg_model,
            "device": cfg_device,
            "batch_size": cfg_batch_size,
            "fp16": cfg_fp16,
            "chunk_size": cfg_chunk_size,
            "chunk_overlap": cfg_overlap,
            "chunk_strategy": cfg_strategy,
            "storage_format": storage_format,
            "runs_per_size": runs,
        },
        "results": all_results,
    }
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="raglet latency benchmark")
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Iterations per size point (default: 20)",
    )
    parser.add_argument(
        "--format",
        choices=["sqlite", "directory"],
        default="sqlite",
        help="Storage format to benchmark (default: sqlite)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/latency/results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--sizes",
        type=float,
        nargs="+",
        default=None,
        help="Text sizes to test in MB (default: 0.005 0.01 0.05 0.1 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in tokens (default: 256)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )

    args = parser.parse_args()
    try:
        run_benchmark(
            runs=args.runs,
            storage_format=args.format,
            output_json=args.output,
            sizes_mb=args.sizes,
            chunk_size=args.chunk_size,
            model=args.model,
        )
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
