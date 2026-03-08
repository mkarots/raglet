# Storage Format Performance Test

## Purpose

End-to-end performance test for the full raglet pipeline (`from_files` → `search` → `save` → `load`) across text sizes and storage formats. Compares SQLite vs. directory format at different scales.

## Method

1. Generate test files at configurable sizes (default: 0.1 MB, 1.0 MB, 10.0 MB)
2. For each size and format combination, measure:
   - `from_files` — full build pipeline (extract → chunk → embed → index)
   - `search` — single query, top_k=5
   - `save` — write to disk
   - `load` — read from disk and rebuild index
3. All timings via `time.perf_counter()`
4. Report file size on disk for each format

## Technology

- **Storage formats:** SQLite (`.sqlite`) and directory (`.raglet/`)
- **Embedding model:** `all-MiniLM-L6-v2`
- **Timing:** `time.perf_counter()`

## Usage

```bash
uv run python benchmarks/storage-performance/run.py
uv run python benchmarks/storage-performance/run.py --sizes 0.1 1.0 5.0 --formats sqlite
```

## Results

No persisted results yet. Run the benchmark to generate `performance_results.json` in the working directory.
