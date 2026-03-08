"""raglet_measure.py — Focused wall-time + memory measurement for every raglet operation.

Unlike raglet_profile.py (which uses cProfile for deep-dives), this script measures:
  • wall time (ms)
  • peak RSS delta (MB) via psutil
  • derived metrics (throughput, percentiles)

for each of the four operation categories:
  1. Startup   — import cost and model-load cost (via subprocess for cold measurement)
  2. Embed     — generate() at multiple chunk counts
  3. Save/Load — all six backend operations (sqlite + directory, full + incr)
  4. Search    — p50 / p95 / p99 latency, separate embed vs FAISS costs

Usage
-----
    uv run python performance/raglet_measure.py
    uv run python performance/raglet_measure.py --chunks 5000
    uv run python performance/raglet_measure.py --skip-startup
    uv run python performance/raglet_measure.py --ops embed,search
    uv run python performance/raglet_measure.py --no-memory

Output
------
    Console: formatted tables after each section
    performance/measurements.json: machine-readable snapshot
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── psutil for RSS ───────────────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ── raglet imports (import order matters on macOS for OpenMP/FAISS) ──────────
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

try:
    import raglet as _raglet_pkg  # noqa: F401
    from raglet import RAGlet
    from raglet.config.config import EmbeddingConfig, RAGletConfig
    from raglet.core.chunk import Chunk
    from raglet.embeddings.generator import SentenceTransformerGenerator
    from raglet.storage.directory_backend import DirectoryStorageBackend
    from raglet.storage.sqlite_backend import SQLiteStorageBackend
    from raglet.vector_store.faiss_store import FAISSVectorStore
except ImportError as exc:
    print(f"Import error: {exc}")
    print("Run from the repo root:  uv run python performance/raglet_measure.py")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────────
BENCH_DIR  = Path("benchmarks/measure-suite")
TEST_DATA  = BENCH_DIR / "test_data"
OUT_FILE   = BENCH_DIR / "results.json"

W = 80  # console width


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    if not _PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1_000_000


def measure(fn: Callable, track_memory: bool = True) -> tuple[float, float]:
    """Run fn() and return (wall_ms, rss_delta_mb).

    rss_delta is the RSS increase — negative deltas (GC freed memory) are
    clamped to 0 because we care about peak allocation, not net change.
    """
    rss_before = _rss_mb() if track_memory else 0.0
    t0 = time.perf_counter()
    fn()
    wall_ms = (time.perf_counter() - t0) * 1000
    rss_after = _rss_mb() if track_memory else 0.0
    return wall_ms, max(0.0, rss_after - rss_before)


def disk_mb(path: str | Path) -> float:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1_000_000
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1_000_000
    return 0.0


def make_chunks(n: int) -> list[Chunk]:
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
            metadata={"chunk_id": i, "doc_id": i // 100},
        )
        for i in range(n)
    ]


def section(title: str) -> None:
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def row(label: str, wall_ms: float, rss_mb: float,
        extra: str = "", mem: bool = True) -> None:
    rss_col = f"+{rss_mb:>5.1f}MB" if mem and _PSUTIL else "       "
    print(f"  {label:<38}  {wall_ms:>8.1f}ms  {rss_col}  {extra}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Startup
# ─────────────────────────────────────────────────────────────────────────────

def _subprocess_time(code: str) -> float:
    """Time a snippet of code in a fresh Python subprocess (ms)."""
    script = (
        "import time as _t; _s = _t.perf_counter(); "
        f"{code}; "
        "print(_t.perf_counter() - _s)"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    try:
        return float(result.stdout.strip()) * 1000
    except (ValueError, AttributeError):
        return float("nan")


def run_startup(sqlite_fixture: Optional[str]) -> dict:
    section("1 · STARTUP  (cold, measured in subprocess)")
    print(f"  {'Operation':<38}  {'wall':>10}  {'note'}")
    print(f"  {'─'*38}  {'─'*10}  {'─'*30}")

    results = {}

    t = _subprocess_time("import raglet")
    results["import_ms"] = round(t, 1)
    print(f"  {'import raglet':<38}  {t:>8.1f}ms  pure import cost")

    t = _subprocess_time(
        "import raglet; from raglet import RAGlet; "
        "from raglet.config.config import RAGletConfig; "
        "RAGlet(chunks=[], config=RAGletConfig())"
    )
    results["model_load_ms"] = round(t, 1)
    print(f"  {'RAGlet() — import + model load':<38}  {t:>8.1f}ms  includes SentenceTransformer.__init__")

    if sqlite_fixture and Path(sqlite_fixture).exists():
        code = (
            f"import raglet; from raglet import RAGlet; "
            f"RAGlet.load('{sqlite_fixture}')"
        )
        t = _subprocess_time(code)
        results["load_raglet_ms"] = round(t, 1)
        print(f"  {'RAGlet.load(sqlite)':<38}  {t:>8.1f}ms  import + load + FAISS rebuild")
    else:
        print(f"  {'RAGlet.load(sqlite)':<38}  {'n/a':>10}  (fixture not found)")
        results["load_raglet_ms"] = None

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Embed
# ─────────────────────────────────────────────────────────────────────────────

def run_embed(gen: SentenceTransformerGenerator, chunk_sizes: list[int],
              track_memory: bool) -> list[dict]:
    section("2 · EMBED  — SentenceTransformerGenerator.generate()")
    print(f"  {'chunks':>8}  {'wall':>10}  {'MB':>8}  {'chunks/s':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}")

    results = []
    for n in chunk_sizes:
        chunks = make_chunks(n)
        wall_ms, rss = measure(lambda c=chunks: gen.generate(c), track_memory)
        cps = n / (wall_ms / 1000) if wall_ms > 0 else 0
        rss_col = f"+{rss:>5.1f}MB" if track_memory and _PSUTIL else "       "
        print(f"  {n:>8,}  {wall_ms:>8.1f}ms  {rss_col}  {cps:>8.0f}/s")
        results.append({
            "n_chunks": n,
            "wall_ms": round(wall_ms, 1),
            "rss_delta_mb": round(rss, 1),
            "chunks_per_sec": round(cps, 0),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def _build_raglet(n: int, gen: SentenceTransformerGenerator,
                  config: RAGletConfig) -> RAGlet:
    """Build an in-memory RAGlet from fresh chunks (used when no fixture)."""
    chunks = make_chunks(n)
    embeddings = gen.generate(chunks)
    store = FAISSVectorStore(embeddings.shape[1], config.search)
    store.add_vectors(embeddings.copy(), chunks)
    return RAGlet(chunks=chunks, config=config,
                  embedding_generator=gen, vector_store=store,
                  embeddings=embeddings)


def _load_raglet_from_fixture(path: str, backend) -> RAGlet:
    return backend.load(path)


def run_storage(gen: SentenceTransformerGenerator, config: RAGletConfig,
                n_chunks: int, track_memory: bool) -> list[dict]:
    section("3 · SAVE / LOAD  (wall time + RSS + disk size)")
    print(f"  {'operation':<24}  {'backend':<12}  {'wall':>10}  {'RSS':>8}  {'disk':>8}")
    print(f"  {'─'*24}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}")

    results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="raglet_measure_"))

    try:
        # ── Build / obtain a loaded RAGlet ───────────────────────────────
        sqlite_fixture = str(TEST_DATA / f"test_{n_chunks}.sqlite")
        dir_fixture    = str(TEST_DATA / f"test_{n_chunks}_dir")
        has_sqlite = Path(sqlite_fixture).exists()
        has_dir    = Path(dir_fixture).exists()

        # Prefer fixture (avoids re-embedding); fallback to fresh build
        if has_sqlite:
            rl = SQLiteStorageBackend().load(sqlite_fixture)
        elif has_dir:
            rl = DirectoryStorageBackend().load(dir_fixture)
        else:
            print(f"  [building {n_chunks}-chunk raglet — no fixture found]")
            rl = _build_raglet(n_chunks, gen, config)

        extra_chunks = make_chunks(50)
        for c in extra_chunks:
            c.index += n_chunks
        extra_embs = gen.generate(extra_chunks)

        def _op(label: str, backend_name: str, fn: Callable,
                disk_path: Optional[str] = None) -> dict:
            wall_ms, rss = measure(fn, track_memory)
            d = disk_mb(disk_path) if disk_path else 0.0
            rss_col = f"+{rss:>5.1f}MB" if track_memory and _PSUTIL else "       "
            disk_col = f"{d:>6.1f}MB" if disk_path else "      —"
            print(f"  {label:<24}  {backend_name:<12}  {wall_ms:>8.1f}ms  "
                  f"{rss_col}  {disk_col}")
            return {"op": f"{backend_name}_{label.replace(' ', '_')}",
                    "wall_ms": round(wall_ms, 1),
                    "rss_delta_mb": round(rss, 1),
                    "disk_mb": round(d, 2)}

        # ── SQLite ───────────────────────────────────────────────────────
        sq_path = str(tmpdir / "bench.sqlite")
        sq_be   = SQLiteStorageBackend()

        r = _op("full save", "sqlite",
                lambda: sq_be.save(rl, sq_path, incremental=False),
                disk_path=sq_path)
        results.append(r)

        r = _op("full load", "sqlite",
                lambda: sq_be.load(sq_path))
        results.append(r)

        r = _op("incr add (50)", "sqlite",
                lambda: sq_be.add_chunks(sq_path, extra_chunks, extra_embs))
        results.append(r)

        # ── Directory ────────────────────────────────────────────────────
        dir_path = str(tmpdir / "bench.raglet")
        dir_be   = DirectoryStorageBackend()

        r = _op("full save", "directory",
                lambda: dir_be.save(rl, dir_path, incremental=False),
                disk_path=dir_path)
        results.append(r)

        r = _op("full load", "directory",
                lambda: dir_be.load(dir_path))
        results.append(r)

        # Build a raglet with n+50 chunks to trigger incremental path
        all_chunks = rl.chunks + extra_chunks
        all_embs   = np.vstack([rl.embeddings, extra_embs])
        store2 = FAISSVectorStore(all_embs.shape[1], config.search)
        store2.add_vectors(all_embs.copy(), all_chunks)
        rl2 = RAGlet(chunks=all_chunks, config=config,
                     embedding_generator=gen, vector_store=store2,
                     embeddings=all_embs)

        r = _op("incr add (50)", "directory",
                lambda: dir_be.save(rl2, dir_path, incremental=True),
                disk_path=dir_path)
        results.append(r)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(gen: SentenceTransformerGenerator, config: RAGletConfig,
               n_chunks: int, n_queries: int, top_k: int) -> dict:
    section(f"4 · SEARCH  ({n_queries} queries, top_k={top_k})")

    # Load or build raglet
    sqlite_fixture = str(TEST_DATA / f"test_{n_chunks}.sqlite")
    if Path(sqlite_fixture).exists():
        rl = SQLiteStorageBackend().load(sqlite_fixture)
    else:
        rl = _build_raglet(n_chunks, gen, config)

    query_texts = [
        f"retrieval augmented generation query number {i}" for i in range(n_queries)
    ]

    embed_times: list[float] = []
    faiss_times: list[float] = []
    total_times: list[float] = []

    for q in query_texts:
        # Embed phase
        t0 = time.perf_counter()
        qvec = gen.generate_single(q)
        embed_ms = (time.perf_counter() - t0) * 1000

        # FAISS phase
        t1 = time.perf_counter()
        rl.vector_store.search(qvec, top_k)
        faiss_ms = (time.perf_counter() - t1) * 1000

        embed_times.append(embed_ms)
        faiss_times.append(faiss_ms)
        total_times.append(embed_ms + faiss_ms)

    def pct(arr: list[float], p: float) -> float:
        return float(np.percentile(arr, p))

    total_wall = sum(total_times) / 1000  # seconds
    qps = n_queries / total_wall if total_wall > 0 else 0

    print(f"\n  {'metric':<36}  {'p50':>8}  {'p95':>8}  {'p99':>8}")
    print(f"  {'─'*36}  {'─'*8}  {'─'*8}  {'─'*8}")

    def prow(label: str, times: list[float]) -> None:
        print(f"  {label:<36}  "
              f"{pct(times,50):>6.1f}ms  "
              f"{pct(times,95):>6.1f}ms  "
              f"{pct(times,99):>6.1f}ms")

    prow("query embed  (generate_single)", embed_times)
    prow("faiss  (index.search)", faiss_times)
    prow("total per query", total_times)
    print(f"\n  throughput: {qps:.1f} queries/sec  "
          f"(embed dominates — MPS/CUDA reduces embed ~5-15×)")

    return {
        "n_queries": n_queries,
        "top_k": top_k,
        "embed_p50_ms": round(pct(embed_times, 50), 2),
        "embed_p95_ms": round(pct(embed_times, 95), 2),
        "embed_p99_ms": round(pct(embed_times, 99), 2),
        "faiss_p50_ms": round(pct(faiss_times, 50), 3),
        "faiss_p95_ms": round(pct(faiss_times, 95), 3),
        "faiss_p99_ms": round(pct(faiss_times, 99), 3),
        "total_p50_ms": round(pct(total_times, 50), 2),
        "total_p95_ms": round(pct(total_times, 95), 2),
        "total_p99_ms": round(pct(total_times, 99), 2),
        "queries_per_sec": round(qps, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="raglet wall-time + memory measurement suite"
    )
    p.add_argument("--chunks", type=int, default=5000,
                   help="Primary chunk count for save/load/search (default: 5000)")
    p.add_argument("--embed-sizes", type=str, default="100,1000,5000",
                   help="Comma-separated chunk counts for embed section (default: 100,1000,5000)")
    p.add_argument("--search-queries", type=int, default=100,
                   help="Number of search queries to run (default: 100)")
    p.add_argument("--top-k", type=int, default=10,
                   help="top_k for search (default: 10)")
    p.add_argument("--ops", type=str, default="all",
                   help="Comma-separated ops: startup,embed,storage,search  (default: all)")
    p.add_argument("--skip-startup", action="store_true",
                   help="Skip startup section (subprocess calls add ~30s)")
    p.add_argument("--no-memory", action="store_true",
                   help="Skip RSS measurement (useful if psutil unavailable)")
    p.add_argument("--out", type=str, default=str(OUT_FILE),
                   help=f"Output JSON path (default: {OUT_FILE})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    track_memory = not args.no_memory and _PSUTIL

    if not _PSUTIL and not args.no_memory:
        print("⚠  psutil not found — memory tracking disabled. "
              "Install with: pip install psutil")

    ops: set[str]
    if args.skip_startup:
        ops = {"embed", "storage", "search"}
    elif args.ops == "all":
        ops = {"startup", "embed", "storage", "search"}
    else:
        ops = set(args.ops.split(","))

    embed_sizes = [int(x) for x in args.embed_sizes.split(",")]
    n = args.chunks

    # ── Shared setup (not measured) ─────────────────────────────────────────
    config = RAGletConfig()

    print(f"\n{'═'*W}")
    print(f"  raglet measurement suite  |  chunks={n:,}  "
          f"device={config.embedding.device}  model={config.embedding.model}")
    print(f"  memory tracking: {'yes (psutil RSS)' if track_memory else 'disabled'}")
    print(f"{'═'*W}")

    print("\n  [setup] Loading embedding model …", end="", flush=True)
    gen = SentenceTransformerGenerator(config.embedding)
    print(" done")

    snapshot: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_chunks": n,
        "device": config.embedding.device,
        "model": config.embedding.model,
        "python": sys.version.split()[0],
        "memory_tracked": track_memory,
    }

    sqlite_fixture = str(TEST_DATA / f"test_{n}.sqlite")

    # ── Run sections ─────────────────────────────────────────────────────────
    if "startup" in ops:
        snapshot["startup"] = run_startup(
            sqlite_fixture if Path(sqlite_fixture).exists() else None
        )

    if "embed" in ops:
        snapshot["embed"] = run_embed(gen, embed_sizes, track_memory)

    if "storage" in ops:
        snapshot["storage"] = run_storage(gen, config, n, track_memory)

    if "search" in ops:
        snapshot["search"] = run_search(
            gen, config, n, args.search_queries, args.top_k
        )

    # ── Write JSON ───────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n{'═'*W}")
    print(f"  Measurements written → {out_path}")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
