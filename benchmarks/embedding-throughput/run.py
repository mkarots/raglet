"""Embedding throughput benchmark: chunks/sec across text sizes.

Isolates the embedding generation step and measures throughput at each
size point, annotated with the equivalent LLM token count.  This shows
the one-time cost of building a raglet vs. the ongoing cost of stuffing
everything into an LLM context window.

Usage:
    uv run python benchmarks/embedding-throughput/run.py
    uv run python benchmarks/embedding-throughput/run.py --runs 3 --chunk-size 256
    uv run python benchmarks/embedding-throughput/run.py --sizes 0.1 1.0 5.0 --model all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

from raglet.config.config import RAGletConfig
from raglet.embeddings.generator import SentenceTransformerGenerator
from raglet.processing.chunker import SentenceAwareChunker

CHARS_PER_LLM_TOKEN = 4

DEFAULT_SIZES_MB = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


def _sizes_to_spec(sizes_mb: list[float]) -> list[dict[str, Any]]:
    """Convert a list of MB values to the internal size spec format."""
    specs = []
    for mb in sizes_mb:
        size_bytes = int(mb * 1024 * 1024)
        if mb < 1.0:
            label = f"{size_bytes / 1024:.0f} KB"
        else:
            label = f"{mb:.0f} MB"
        specs.append({"label": label, "bytes": size_bytes})
    return specs

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


def generate_text(target_bytes: int) -> str:
    """Return a block of realistic prose of approximately *target_bytes*."""
    para_bytes = len(PARAGRAPH.encode())
    repeats = max(1, target_bytes // para_bytes)
    return PARAGRAPH * repeats


def run_benchmark(
    runs: int = 5,
    output_json: str = "benchmarks/embedding-throughput/results.json",
    sizes_mb: list[float] | None = None,
    chunk_size: int | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Measure embedding throughput at each size point.

    Args:
        runs: Number of iterations per size point.
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
    chunker = SentenceAwareChunker(config.chunking)
    text_sizes = _sizes_to_spec(sizes_mb or DEFAULT_SIZES_MB)

    print("Loading embedding model …", flush=True)
    generator = SentenceTransformerGenerator(config.embedding)
    device = config.embedding.device
    model_name = config.embedding.model
    batch_size = config.embedding.batch_size
    fp16 = config.embedding.use_fp16
    chunk_size = config.chunking.size
    chunk_overlap = config.chunking.overlap
    chunk_strategy = config.chunking.strategy
    max_seq_length = generator.model.max_seq_length

    print(f"  model: {model_name}")
    print(f"  device: {device}")
    print(f"  batch_size: {batch_size}")
    print(f"  fp16: {fp16}")
    print(f"  max_seq_length: {max_seq_length}")
    print(f"  chunk_size: {chunk_size} tokens (~{chunk_size * 4} chars, overlap: {chunk_overlap}, strategy: {chunk_strategy})")

    all_results: list[dict[str, Any]] = []

    print(f"\n{'=' * 105}")
    print(f"Embedding throughput — {runs} runs per size")
    print(f"{'=' * 105}")

    try:
        for size_info in text_sizes:
            label = size_info["label"]
            target = size_info["bytes"]
            approx_tokens = target // CHARS_PER_LLM_TOKEN

            text = generate_text(target)
            chunks = chunker.chunk(text, metadata={"source": "bench"})
            n_chunks = len(chunks)

            print(f"\n{'─' * 105}")
            print(f"  {label}  (~{approx_tokens:,} LLM tokens, {n_chunks} chunks)")
            print(f"{'─' * 105}")

            durations: list[float] = []
            for r in range(runs):
                gc.collect()
                t0 = time.perf_counter()
                _emb = generator.generate(chunks)
                elapsed = time.perf_counter() - t0
                durations.append(elapsed)
                del _emb

            throughputs = [n_chunks / d for d in durations]
            mean_dur = sum(durations) / len(durations)
            mean_tp = sum(throughputs) / len(throughputs)
            min_tp = min(throughputs)
            max_tp = max(throughputs)

            avg_chunk_chars = sum(len(c.text) for c in chunks) / max(n_chunks, 1)

            row: dict[str, Any] = {
                "text_size": label,
                "text_bytes": target,
                "approx_llm_tokens": approx_tokens,
                "chunks": n_chunks,
                "avg_chunk_chars": round(avg_chunk_chars),
                "mean_embed_s": round(mean_dur, 3),
                "mean_chunks_per_s": round(mean_tp, 1),
                "min_chunks_per_s": round(min_tp, 1),
                "max_chunks_per_s": round(max_tp, 1),
                "llm_tokens_per_s": round(mean_tp * avg_chunk_chars / CHARS_PER_LLM_TOKEN),
            }
            all_results.append(row)

            print(
                f"  {mean_dur:.3f}s  "
                f"{mean_tp:,.0f} chunks/s  "
                f"(min {min_tp:,.0f}, max {max_tp:,.0f})  "
                f"~{row['llm_tokens_per_s']:,} LLM tokens/s ingested"
            )

    finally:
        generator.close()

    # ── Summary table ──
    print(f"\n{'=' * 105}")
    print("SUMMARY")
    print(f"{'=' * 105}")

    hdr = (
        f"{'Size':>8} {'~LLM tokens':>14} {'Chunks':>8} │"
        f" {'embed time':>11} {'chunks/s':>10} {'min':>8} {'max':>8} │"
        f" {'LLM tok/s ingested':>20}"
    )
    print(hdr)
    print("─" * len(hdr))

    for r in all_results:
        tok_str = f"{r['approx_llm_tokens']:,}"
        tok_per_s = f"{r['llm_tokens_per_s']:,}"
        print(
            f"{r['text_size']:>8} {tok_str:>14} {r['chunks']:>8,} │"
            f" {r['mean_embed_s']:>10.3f}s {r['mean_chunks_per_s']:>10,.0f}"
            f" {r['min_chunks_per_s']:>8,.0f} {r['max_chunks_per_s']:>8,.0f} │"
            f" {tok_per_s:>20}"
        )

    print(
        f"\nModel: {model_name} on {device} (batch_size={batch_size}, fp16={fp16})"
    )
    print(
        f"Chunks: {chunk_size} tokens (~{chunk_size * 4} chars), overlap {chunk_overlap} tokens, strategy {chunk_strategy}"
    )
    print(
        "Embedding is a one-time build cost. After building, search is <10 ms "
        "regardless of dataset size."
    )

    output_data = {
        "config": {
            "model": model_name,
            "device": device,
            "batch_size": batch_size,
            "fp16": fp16,
            "max_seq_length": max_seq_length,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_strategy": chunk_strategy,
            "runs_per_size": runs,
        },
        "results": all_results,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="raglet embedding throughput benchmark")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Iterations per size point (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/embedding-throughput/results.json",
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
            output_json=args.output,
            sizes_mb=args.sizes,
            chunk_size=args.chunk_size,
            model=args.model,
        )
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
