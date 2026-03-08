"""Targeted cProfile analysis for raglet operations.

Profiles each raglet operation independently so torch/sentence-transformers
noise does not obscure the raglet-owned hotspots.

Operations profiled
-------------------
  1. embed          – SentenceTransformerGenerator.generate()
  2. faiss_add      – FAISSVectorStore.add_vectors()
  3. sqlite_save    – SQLiteStorageBackend.save() full write
  4. sqlite_load    – SQLiteStorageBackend.load() incl. FAISS rebuild
  5. sqlite_incr    – SQLiteStorageBackend.add_chunks() incremental
  6. dir_save       – DirectoryStorageBackend.save() full write
  7. dir_load       – DirectoryStorageBackend.load() incl. FAISS rebuild
  8. dir_incr       – DirectoryStorageBackend incremental save
  9. search         – RAGlet.search() × N queries

Usage
-----
    uv run python benchmarks/cprofile-analysis/run.py
    uv run python benchmarks/cprofile-analysis/run.py --size 5000
    uv run python benchmarks/cprofile-analysis/run.py --size 10000 --search-queries 50
    uv run python benchmarks/cprofile-analysis/run.py --ops embed,sqlite_save,sqlite_load

Output
------
  Console: per-operation tables (raglet-filtered and full top-10)
  benchmarks/cprofile-analysis/profile_results/  – one .prof file per operation
  benchmarks/cprofile-analysis/profile_results/summary.json  – machine-readable timings
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Import order is load-bearing on macOS (OpenMP / FAISS) ──────────────────
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

try:
    import raglet  # noqa: F401  triggers correct torch→faiss init order
    from raglet import RAGlet
    from raglet.config.config import EmbeddingConfig, RAGletConfig
    from raglet.core.chunk import Chunk
    from raglet.embeddings.generator import SentenceTransformerGenerator
    from raglet.storage.directory_backend import DirectoryStorageBackend
    from raglet.storage.sqlite_backend import SQLiteStorageBackend
    from raglet.vector_store.faiss_store import FAISSVectorStore
except ImportError as exc:
    print(f"Import error: {exc}")
    print("Run from the repo root:  uv run python benchmarks/cprofile-analysis/run.py")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
RAGLET_SRC_MARKER = "raglet"          # path fragment that marks our code
STDLIB_MARKERS = ("json", "sqlite3", "numpy", "zipfile", "io", "pathlib")
OUTPUT_DIR = Path("benchmarks/cprofile-analysis/profile_results")

COL_W = 70   # function name column width
SEP = "─" * (COL_W + 30)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_chunks(n: int) -> List[Chunk]:
    base = (
        "Retrieval-augmented generation stores text chunks alongside vector "
        "embeddings so that a similarity search can surface the most relevant "
        "passages for a given query. This chunk represents realistic content."
    )
    return [
        Chunk(
            text=f"{base} Chunk {i}.",
            source=f"doc_{i // 100}.md",
            index=i,
            metadata={"chunk_id": i, "doc_id": i // 100, "section": "body"},
        )
        for i in range(n)
    ]


def warm_model(config: EmbeddingConfig) -> SentenceTransformerGenerator:
    """Load and warm up the embedding model (not profiled)."""
    print("  Warming up embedding model …", end="", flush=True)
    gen = SentenceTransformerGenerator(config)
    gen.generate([Chunk(text="warmup", source="w", index=0, metadata={})])
    print(" done")
    return gen


def _is_raglet_or_stdlib(filename: str) -> bool:
    if RAGLET_SRC_MARKER in filename:
        return True
    return any(m in filename for m in STDLIB_MARKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Profiling helpers
# ─────────────────────────────────────────────────────────────────────────────

class OpResult:
    """Holds timing and profile stats for one operation."""

    def __init__(self, name: str, wall_s: float, profiler: cProfile.Profile):
        self.name = name
        self.wall_s = wall_s
        self.profiler = profiler
        self._stats = pstats.Stats(profiler, stream=io.StringIO())
        self._stats.sort_stats("cumulative")

    # ── extract top-N rows from pstats ──────────────────────────────────────
    def top_rows(self, n: int = 12, raglet_only: bool = False) -> List[dict]:
        rows = []
        for func, (cc, nc, tt, ct, _callers) in self._stats.stats.items():
            fname, lineno, funcname = func
            if raglet_only and not _is_raglet_or_stdlib(fname):
                continue
            rows.append(
                dict(
                    file=fname,
                    line=lineno,
                    func=funcname,
                    calls=nc,
                    tottime=tt,
                    cumtime=ct,
                )
            )
        rows.sort(key=lambda r: r["cumtime"], reverse=True)
        return rows[:n]

    def total_cpu(self) -> float:
        return self._stats.total_tt

    def dump(self, path: Path) -> None:
        self.profiler.dump_stats(str(path))


def _run(fn, name: str) -> OpResult:
    """Profile a zero-argument callable and return an OpResult."""
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.runcall(fn)
    wall = time.perf_counter() - t0
    return OpResult(name, wall, profiler)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_func(file: str, func: str, max_w: int = COL_W) -> str:
    # Show only the relevant path suffix
    parts = file.replace("\\", "/").split("/")
    # Keep from "raglet/" or "site-packages/…" onward
    try:
        idx = next(i for i, p in enumerate(parts) if p == "raglet")
        short = "/".join(parts[idx:])
    except StopIteration:
        short = "/".join(parts[-3:])
    label = f"{short}  {func}()"
    return label[:max_w] if len(label) > max_w else label


def _print_table(result: OpResult, raglet_only: bool = False, n: int = 12) -> None:
    tag = "raglet + stdlib" if raglet_only else "all (top)"
    print(f"\n  ┌─ {result.name}  [{tag}]")
    print(f"  │  wall={result.wall_s*1000:.1f}ms   cpu={result.total_cpu()*1000:.1f}ms")
    print(f"  │  {'Function':<{COL_W}}  {'calls':>7}  {'tottime':>9}  {'cumtime':>9}  {'%wall':>7}")
    print(f"  │  {'─'*COL_W}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*7}")
    for r in result.top_rows(n=n, raglet_only=raglet_only):
        label = _fmt_func(r["file"], r["func"])
        pct = r["cumtime"] / result.wall_s * 100 if result.wall_s > 0 else 0
        print(
            f"  │  {label:<{COL_W}}  {r['calls']:>7}  "
            f"{r['tottime']*1000:>8.2f}ms  {r['cumtime']*1000:>8.2f}ms  {pct:>6.1f}%"
        )
    print(f"  └{'─'*80}")


def print_section(title: str) -> None:
    w = 80
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual operation profilers
# ─────────────────────────────────────────────────────────────────────────────

def profile_embed(chunks: List[Chunk], gen: SentenceTransformerGenerator) -> OpResult:
    def fn():
        gen.generate(chunks)
    return _run(fn, "embed")


def profile_faiss_add(embeddings: np.ndarray, chunks: List[Chunk], config: RAGletConfig) -> OpResult:
    store = FAISSVectorStore(embeddings.shape[1], config.search)
    def fn():
        store.add_vectors(embeddings.copy(), chunks)
    return _run(fn, "faiss_add")


def profile_sqlite_save(raglet_inst: RAGlet, path: str) -> OpResult:
    backend = SQLiteStorageBackend()
    if Path(path).exists():
        os.remove(path)
    def fn():
        backend.save(raglet_inst, path, incremental=False)
    return _run(fn, "sqlite_save")


def profile_sqlite_load(path: str) -> OpResult:
    backend = SQLiteStorageBackend()
    def fn():
        backend.load(path)
    return _run(fn, "sqlite_load")


def profile_sqlite_incr(
    path: str,
    extra_chunks: List[Chunk],
    extra_embeddings: np.ndarray,
) -> OpResult:
    backend = SQLiteStorageBackend()
    def fn():
        backend.add_chunks(path, extra_chunks, extra_embeddings)
    return _run(fn, "sqlite_incr")


def profile_dir_save(raglet_inst: RAGlet, path: str) -> OpResult:
    backend = DirectoryStorageBackend()
    if Path(path).exists():
        shutil.rmtree(path)
    def fn():
        backend.save(raglet_inst, path, incremental=False)
    return _run(fn, "dir_save")


def profile_dir_load(path: str) -> OpResult:
    backend = DirectoryStorageBackend()
    def fn():
        backend.load(path)
    return _run(fn, "dir_load")


def profile_dir_incr(raglet_inst: RAGlet, path: str) -> OpResult:
    """Profile incremental directory save."""
    backend = DirectoryStorageBackend()
    def fn():
        backend.save(raglet_inst, path, incremental=True)
    return _run(fn, "dir_incr")


def profile_search(raglet_inst: RAGlet, queries: List[str], top_k: int = 10) -> OpResult:
    def fn():
        for q in queries:
            raglet_inst.search(q, top_k=top_k)
    return _run(fn, f"search×{len(queries)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[OpResult], n_chunks: int, n_search: int) -> dict:
    """Print final comparison table and return JSON-serialisable dict."""
    print_section("SUMMARY — wall-clock time per operation")
    print(f"\n  Chunk count : {n_chunks:,}")
    print(f"  Search queries : {n_search}\n")
    print(f"  {'Operation':<22}  {'Wall (ms)':>10}  {'CPU (ms)':>10}  {'Note'}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*40}")

    summary = {"n_chunks": n_chunks, "operations": []}
    for r in results:
        note = ""
        if r.name == "embed":
            note = "⚡ use MPS/CUDA to cut 5-15×"
        elif r.name == "faiss_add":
            note = "normalize_L2 + index.add"
        elif "save" in r.name:
            note = "serialise + write to disk"
        elif "load" in r.name:
            note = "deserialise + FAISS rebuild"
        elif "incr" in r.name:
            note = "append-only write"
        elif "search" in r.name:
            note = f"avg {r.wall_s/max(n_search,1)*1000:.2f}ms/query"

        print(f"  {r.name:<22}  {r.wall_s*1000:>10.1f}  {r.total_cpu()*1000:>10.1f}  {note}")
        summary["operations"].append(
            {
                "name": r.name,
                "wall_ms": round(r.wall_s * 1000, 2),
                "cpu_ms": round(r.total_cpu() * 1000, 2),
                "top_raglet_hotspots": [
                    {
                        "func": h["func"],
                        "cumtime_ms": round(h["cumtime"] * 1000, 2),
                        "pct_wall": round(h["cumtime"] / r.wall_s * 100, 1) if r.wall_s > 0 else 0,
                    }
                    for h in r.top_rows(n=5, raglet_only=True)
                ],
            }
        )

    print()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted raglet cProfile analysis")
    p.add_argument("--size", type=int, default=2000,
                   help="Number of chunks to use (default: 2000)")
    p.add_argument("--search-queries", type=int, default=20,
                   help="Number of search queries to run (default: 20)")
    p.add_argument("--ops", type=str, default="all",
                   help=(
                       "Comma-separated operations to run "
                       "(default: all). Options: "
                       "embed, faiss_add, sqlite_save, sqlite_load, sqlite_incr, "
                       "dir_save, dir_load, dir_incr, search"
                   ))
    p.add_argument("--no-raglet-filter", action="store_true",
                   help="Show full cProfile output instead of raglet-filtered view")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ops_requested = (
        set(args.ops.split(",")) if args.ops != "all"
        else {
            "embed", "faiss_add",
            "sqlite_save", "sqlite_load", "sqlite_incr",
            "dir_save", "dir_load", "dir_incr",
            "search",
        }
    )
    raglet_filter = not args.no_raglet_filter
    n = args.size
    n_search = args.search_queries

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="raglet_prof_"))

    print_section(f"raglet targeted profile  |  size={n:,}  search={n_search}")

    # ── 1. Setup (not profiled) ──────────────────────────────────────────────
    config = RAGletConfig()
    print("\n[setup] Loading model and generating embeddings for test data …")
    gen = warm_model(config.embedding)

    print(f"[setup] Creating {n:,} test chunks …", end="", flush=True)
    chunks = make_chunks(n)
    print(f" done")

    print("[setup] Pre-generating embeddings (not profiled) …", end="", flush=True)
    t0 = time.perf_counter()
    embeddings = gen.generate(chunks)
    embed_wall = time.perf_counter() - t0
    print(f" {embed_wall*1000:.0f}ms  shape={embeddings.shape}")

    # Build a raglet instance for save operations
    print("[setup] Building in-memory RAGlet …", end="", flush=True)
    store = FAISSVectorStore(embeddings.shape[1], config.search)
    store.add_vectors(embeddings.copy(), chunks)
    import raglet as raglet_module
    rl = RAGlet(chunks=chunks, config=config, embedding_generator=gen,
                vector_store=store, embeddings=embeddings)
    print(" done\n")

    results: List[OpResult] = []

    # ── 2. Embedding generation ──────────────────────────────────────────────
    if "embed" in ops_requested:
        print_section("1 · embed  — SentenceTransformerGenerator.generate()")
        r = profile_embed(chunks, gen)
        results.append(r)
        _print_table(r, raglet_only=raglet_filter)
        _print_table(r, raglet_only=False, n=8)
        r.dump(OUTPUT_DIR / "embed.prof")

    # ── 3. FAISS add_vectors ─────────────────────────────────────────────────
    if "faiss_add" in ops_requested:
        print_section("2 · faiss_add  — FAISSVectorStore.add_vectors()")
        r = profile_faiss_add(embeddings, chunks, config)
        results.append(r)
        _print_table(r, raglet_only=raglet_filter)
        r.dump(OUTPUT_DIR / "faiss_add.prof")

    # ── 4. SQLite save ───────────────────────────────────────────────────────
    sqlite_path = str(tmpdir / "test.sqlite")
    if "sqlite_save" in ops_requested:
        print_section("3 · sqlite_save  — SQLiteStorageBackend.save() full")
        r = profile_sqlite_save(rl, sqlite_path)
        results.append(r)
        size_mb = Path(sqlite_path).stat().st_size / 1_000_000
        print(f"  File size: {size_mb:.2f} MB")
        _print_table(r, raglet_only=raglet_filter)
        r.dump(OUTPUT_DIR / "sqlite_save.prof")
    else:
        # Still need the file for load/incr
        SQLiteStorageBackend().save(rl, sqlite_path, incremental=False)

    # ── 5. SQLite load ───────────────────────────────────────────────────────
    if "sqlite_load" in ops_requested:
        print_section("4 · sqlite_load  — SQLiteStorageBackend.load() + FAISS rebuild")
        r = profile_sqlite_load(sqlite_path)
        results.append(r)
        _print_table(r, raglet_only=raglet_filter)
        r.dump(OUTPUT_DIR / "sqlite_load.prof")

    # ── 6. SQLite incremental ────────────────────────────────────────────────
    if "sqlite_incr" in ops_requested:
        print_section("5 · sqlite_incr  — SQLiteStorageBackend.add_chunks() incremental")
        extra_chunks = make_chunks(50)
        for c in extra_chunks:
            c.index += n
        extra_embs = gen.generate(extra_chunks)
        r = profile_sqlite_incr(sqlite_path, extra_chunks, extra_embs)
        results.append(r)
        _print_table(r, raglet_only=raglet_filter)
        r.dump(OUTPUT_DIR / "sqlite_incr.prof")

    # ── 7. Directory save ────────────────────────────────────────────────────
    dir_path = str(tmpdir / "test_dir.raglet")
    if "dir_save" in ops_requested:
        print_section("6 · dir_save  — DirectoryStorageBackend.save() full")
        r = profile_dir_save(rl, dir_path)
        results.append(r)
        dir_size = sum(f.stat().st_size for f in Path(dir_path).rglob("*") if f.is_file())
        print(f"  Directory size: {dir_size/1_000_000:.2f} MB")
        _print_table(r, raglet_only=raglet_filter)
        r.dump(OUTPUT_DIR / "dir_save.prof")
    else:
        DirectoryStorageBackend().save(rl, dir_path, incremental=False)

    # ── 8. Directory load ────────────────────────────────────────────────────
    if "dir_load" in ops_requested:
        print_section("7 · dir_load  — DirectoryStorageBackend.load() + FAISS rebuild")
        r = profile_dir_load(dir_path)
        results.append(r)
        _print_table(r, raglet_only=raglet_filter)
        r.dump(OUTPUT_DIR / "dir_load.prof")

    # ── 9. Directory incremental ─────────────────────────────────────────────
    if "dir_incr" in ops_requested:
        print_section("8 · dir_incr  — DirectoryStorageBackend incremental save")
        # Build a raglet with n+50 chunks to trigger incremental path
        extra_chunks_2 = make_chunks(50)
        for c in extra_chunks_2:
            c.index += n
        extra_embs_2 = gen.generate(extra_chunks_2)
        all_chunks = chunks + extra_chunks_2
        all_embs = np.vstack([embeddings, extra_embs_2])
        store2 = FAISSVectorStore(all_embs.shape[1], config.search)
        store2.add_vectors(all_embs.copy(), all_chunks)
        rl2 = RAGlet(chunks=all_chunks, config=config, embedding_generator=gen,
                     vector_store=store2, embeddings=all_embs)
        r = profile_dir_incr(rl2, dir_path)
        results.append(r)
        _print_table(r, raglet_only=raglet_filter)
        r.dump(OUTPUT_DIR / "dir_incr.prof")

    # ── 10. Search ───────────────────────────────────────────────────────────
    if "search" in ops_requested:
        print_section(f"9 · search  — RAGlet.search()  ×{n_search} queries")
        query_texts = [
            f"retrieval augmented generation query number {i}"
            for i in range(n_search)
        ]
        r = profile_search(rl, query_texts, top_k=10)
        results.append(r)
        _print_table(r, raglet_only=raglet_filter)
        r.dump(OUTPUT_DIR / f"search.prof")

    # ── Final summary ────────────────────────────────────────────────────────
    summary = print_summary(results, n, n_search)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  .prof files  →  {OUTPUT_DIR}/")
    print(f"  summary      →  {summary_path}")
    print(f"\n  Inspect interactively:")
    print(f"    snakeviz {OUTPUT_DIR}/sqlite_load.prof")
    print(f"    snakeviz {OUTPUT_DIR}/embed.prof\n")

    # Cleanup tmp
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
