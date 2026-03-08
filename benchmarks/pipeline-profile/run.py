#!/usr/bin/env python3
"""Isolated pipeline profiler for raglet build.

Measures each stage of the build pipeline independently:
  1. File discovery (expand_file_inputs)
  2. Text extraction (per-file I/O)
  3. Chunking (sentence-aware splitting)
  4. Embedding generation (sentence-transformers inference)
  5. FAISS indexing (normalize + add)
  6. Save (directory backend serialisation)

Usage:
    uv run python benchmarks/pipeline-profile/run.py data/
    uv run python benchmarks/pipeline-profile/run.py data/ --save-format sqlite
"""

import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _peak_rss_mb() -> float:
    """Current process RSS in MB (macOS/Linux)."""
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return rusage.ru_maxrss / (1024 * 1024)
    return rusage.ru_maxrss / 1024


def profile(input_path: str, save_format: str = "directory") -> None:
    from raglet.config.config import ChunkingConfig, EmbeddingConfig, RAGletConfig, SearchConfig
    from raglet.processing.chunker import SentenceAwareChunker
    from raglet.processing.extractor_factory import create_extractor
    from raglet.utils import expand_file_inputs

    config = RAGletConfig()
    results: dict[str, dict] = {}

    print("=" * 70)
    print(f"Pipeline profile: {input_path}")
    print(f"Save format:      {save_format}")
    print(f"Chunk size:       {config.chunking.size} tokens (~{config.chunking.size * 4} chars)")
    print(f"Overlap:          {config.chunking.overlap} tokens (~{config.chunking.overlap * 4} chars)")
    print(f"Embedding model:  {config.embedding.model}")
    print(f"Device:           {config.embedding.device}")
    print("=" * 70)

    # ── Stage 1: File discovery ──────────────────────────────────────────
    gc.collect()
    t0 = time.perf_counter()
    file_paths = expand_file_inputs([input_path])
    t_discover = time.perf_counter() - t0

    total_bytes = sum(os.path.getsize(f) for f in file_paths)
    results["discovery"] = {"time": t_discover, "files": len(file_paths), "bytes": total_bytes}

    print(f"\n1. File discovery")
    print(f"   Files:  {len(file_paths):,}")
    print(f"   Size:   {_fmt_bytes(total_bytes)}")
    print(f"   Time:   {_fmt_time(t_discover)}")

    # ── Stage 2: Text extraction ─────────────────────────────────────────
    gc.collect()
    texts: list[tuple[str, str]] = []  # (file_path, extracted_text)
    t0 = time.perf_counter()
    for fp in file_paths:
        extractor = create_extractor(fp)
        text = extractor.extract(fp)
        texts.append((fp, text))
    t_extract = time.perf_counter() - t0

    total_text_chars = sum(len(t) for _, t in texts)
    results["extraction"] = {"time": t_extract, "chars": total_text_chars}

    print(f"\n2. Text extraction")
    print(f"   Chars:  {total_text_chars:,} ({_fmt_bytes(total_text_chars)})")
    print(f"   Time:   {_fmt_time(t_extract)}")
    print(f"   Rate:   {total_text_chars / t_extract / 1e6:.1f} M chars/s")

    # ── Stage 3: Chunking ────────────────────────────────────────────────
    gc.collect()
    chunker = SentenceAwareChunker(config.chunking)
    all_chunks = []
    t0 = time.perf_counter()
    for fp, text in texts:
        chunks = chunker.chunk(text, metadata={"source": fp})
        all_chunks.extend(chunks)
    t_chunk = time.perf_counter() - t0

    chunk_text_sizes = [len(c.text) for c in all_chunks]
    results["chunking"] = {
        "time": t_chunk,
        "chunks": len(all_chunks),
        "avg_chars": int(np.mean(chunk_text_sizes)) if chunk_text_sizes else 0,
        "median_chars": int(np.median(chunk_text_sizes)) if chunk_text_sizes else 0,
        "min_chars": min(chunk_text_sizes) if chunk_text_sizes else 0,
        "max_chars": max(chunk_text_sizes) if chunk_text_sizes else 0,
    }

    print(f"\n3. Chunking")
    print(f"   Chunks: {len(all_chunks):,}")
    print(f"   Avg:    {results['chunking']['avg_chars']:,} chars/chunk")
    print(f"   Range:  [{results['chunking']['min_chars']:,} – {results['chunking']['max_chars']:,}] chars")
    print(f"   Time:   {_fmt_time(t_chunk)}")

    # Free raw texts
    del texts
    gc.collect()

    # ── Stage 4: Embedding generation ────────────────────────────────────
    from raglet.embeddings.generator import SentenceTransformerGenerator

    print(f"\n4. Embedding generation")
    print(f"   Loading model...", end="", flush=True)
    t0 = time.perf_counter()
    generator = SentenceTransformerGenerator(config.embedding)
    t_model_load = time.perf_counter() - t0
    print(f" {_fmt_time(t_model_load)}")

    dim = generator.get_dimension()
    expected_bytes = len(all_chunks) * dim * 4  # float32

    gc.collect()
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    embeddings = generator.generate(all_chunks)
    t_embed = time.perf_counter() - t0
    rss_after = _peak_rss_mb()

    results["embedding"] = {
        "time": t_embed,
        "model_load_time": t_model_load,
        "shape": list(embeddings.shape),
        "bytes": embeddings.nbytes,
        "chunks_per_sec": len(all_chunks) / t_embed if t_embed > 0 else 0,
        "rss_delta_mb": rss_after - rss_before,
    }

    print(f"   Shape:  {embeddings.shape} ({_fmt_bytes(embeddings.nbytes)})")
    print(f"   Time:   {_fmt_time(t_embed)}")
    print(f"   Rate:   {results['embedding']['chunks_per_sec']:,.0f} chunks/s")
    print(f"   RSS Δ:  {rss_after - rss_before:+.1f} MB")

    # ── Stage 5: FAISS indexing ──────────────────────────────────────────
    from raglet.vector_store.faiss_store import FAISSVectorStore

    store = FAISSVectorStore(embedding_dim=dim, config=config.search)

    gc.collect()
    emb_copy = embeddings.copy()  # add_vectors normalises in-place
    t0 = time.perf_counter()
    store.add_vectors(emb_copy, all_chunks)
    t_faiss = time.perf_counter() - t0
    del emb_copy

    results["faiss"] = {"time": t_faiss, "count": store.get_count()}

    print(f"\n5. FAISS indexing")
    print(f"   Vectors: {store.get_count():,}")
    print(f"   Time:    {_fmt_time(t_faiss)}")

    # ── Stage 6: Save ────────────────────────────────────────────────────
    from raglet.core.rag import RAGlet

    raglet = RAGlet(
        chunks=all_chunks, config=config,
        embedding_generator=generator, vector_store=store,
        embeddings=embeddings,
    )

    tmpdir = tempfile.mkdtemp(prefix="raglet_profile_")
    if save_format == "sqlite":
        save_path = os.path.join(tmpdir, "output.sqlite")
    else:
        save_path = os.path.join(tmpdir, "output_dir")

    gc.collect()
    t0 = time.perf_counter()
    raglet.save(save_path)
    t_save = time.perf_counter() - t0

    if os.path.isfile(save_path):
        save_bytes = os.path.getsize(save_path)
    else:
        save_bytes = sum(
            f.stat().st_size for f in Path(save_path).rglob("*") if f.is_file()
        )

    results["save"] = {"time": t_save, "bytes": save_bytes, "format": save_format}

    print(f"\n6. Save ({save_format})")
    print(f"   Size:   {_fmt_bytes(save_bytes)}")
    print(f"   Time:   {_fmt_time(t_save)}")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    raglet.close()

    # ── Summary ──────────────────────────────────────────────────────────
    stages = [
        ("Discovery", results["discovery"]["time"]),
        ("Extraction", results["extraction"]["time"]),
        ("Chunking", results["chunking"]["time"]),
        ("Model load", results["embedding"]["model_load_time"]),
        ("Embedding", results["embedding"]["time"]),
        ("FAISS", results["faiss"]["time"]),
        ("Save", results["save"]["time"]),
    ]
    total = sum(t for _, t in stages)

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Stage':<16} {'Time':>10} {'%':>7}")
    print(f"{'-' * 35}")
    for name, t in stages:
        print(f"{name:<16} {_fmt_time(t):>10} {t / total * 100:>6.1f}%")
    print(f"{'-' * 35}")
    print(f"{'TOTAL':<16} {_fmt_time(total):>10} {'100.0%':>7}")
    print(f"\nInput:   {len(file_paths):,} files, {_fmt_bytes(total_bytes)}")
    print(f"Output:  {len(all_chunks):,} chunks, {embeddings.shape[1]}d embeddings")
    print(f"Peak RSS: {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile raglet build pipeline")
    parser.add_argument("input", help="Input path (file, directory, or glob)")
    parser.add_argument(
        "--save-format", choices=["directory", "sqlite"], default="directory",
        help="Save format to benchmark (default: directory)",
    )
    args = parser.parse_args()
    profile(args.input, args.save_format)
