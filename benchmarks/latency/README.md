# Latency Benchmark

## Purpose

Measures p50/p95/p99 latency for the three core operations — **save**, **load**, and **search** — across text sizes from 5 KB to 5 MB. This demonstrates that raglet retrieval gives sub-10ms access to knowledge that would otherwise require stuffing hundreds of thousands of tokens into an LLM context window.

## Method

1. Generate synthetic text files at each size point (5 KB through 5 MB)
2. Build a raglet once per size (not timed in the latency loop)
3. Run N iterations (default: 20) of save → load → search
4. Measure each operation with `time.perf_counter()`
5. Compute p50/p95/p99/mean percentiles for each operation

## Technology

- **Storage format:** SQLite by default (configurable via `--format`)
- **Embedding model:** `all-MiniLM-L6-v2`
- **Search queries:** 5 rotating semantic queries per iteration
- **Timing:** `time.perf_counter()` (high-resolution wall clock)
- **Percentile calculation:** linear interpolation

## Usage

```bash
uv run python benchmarks/latency/run.py
uv run python benchmarks/latency/run.py --runs 30 --format directory
```

## Results

Last run on Apple M1 Pro (SQLite format, 20 iterations per size):

| Size | ~LLM tokens | Chunks | Save p50 | Load p50 | Search p50 |
|-----:|------------:|-------:|---------:|---------:|-----------:|
| 5 KB | 1,280 | 3 | 0.44ms | 5.25ms | 4.86ms |
| 10 KB | 2,560 | 6 | 0.53ms | 4.85ms | 4.42ms |
| 50 KB | 12,800 | 29 | 0.64ms | 4.94ms | 4.34ms |
| 100 KB | 25,600 | 57 | 0.77ms | 4.91ms | 4.23ms |
| 500 KB | 128,000 | 284 | 2.07ms | 5.65ms | 4.44ms |
| 1 MB | 262,144 | 581 | 4.56ms | 7.51ms | 4.93ms |
| 2 MB | 524,288 | 1,162 | 8.03ms | 9.06ms | 5.03ms |
| 5 MB | 1,310,720 | 2,904 | 21.31ms | 18.12ms | 7.14ms |

Search stays under 10ms even at 5 MB / 2,900 chunks. Save and load scale linearly with chunk count.

Raw results: [`results.json`](results.json)
