# Wall-Time + Memory Measurement Suite

## Purpose

Comprehensive measurement suite covering **all four operation categories** with wall-time and memory (RSS) tracking. Unlike the other benchmarks that focus on one aspect, this gives a full picture of raglet's runtime characteristics in a single run.

## Method

Measures four categories:

1. **Startup** — import cost and model-load cost (measured in a fresh subprocess for cold measurement)
2. **Embed** — `SentenceTransformerGenerator.generate()` at multiple chunk counts
3. **Save/Load** — all six backend operations: SQLite + directory, each with full save, full load, and incremental add (50 chunks)
4. **Search** — p50/p95/p99 latency with separate embed vs. FAISS breakdown

Memory is tracked via `psutil.Process().memory_info().rss` (RSS delta per operation).

## Technology

- **Timing:** `time.perf_counter()` (wall clock)
- **Memory:** `psutil` RSS tracking (optional — runs without it, just no memory data)
- **Startup measurement:** subprocess isolation for cold import timing
- **Fixtures:** pre-built raglet fixtures in `test_data/` (optional, falls back to fresh builds)

## Usage

```bash
uv run python benchmarks/measure-suite/run.py
uv run python benchmarks/measure-suite/run.py --chunks 5000
uv run python benchmarks/measure-suite/run.py --ops embed,search
uv run python benchmarks/measure-suite/run.py --skip-startup --no-memory
```

## Results

Output: `benchmarks/measure-suite/results.json` (machine-readable JSON snapshot with timestamps).
