# Embedding Throughput Benchmark

## Purpose

Measures how fast raglet generates embeddings (chunks/second) across text sizes from 5 KB to 5 MB. This isolates the embedding step — the dominant cost in building a raglet — and compares it against the equivalent LLM token count to show the one-time cost of building vs. the ongoing cost of stuffing everything into a context window.

## Method

1. Generate synthetic prose at each size point (5 KB, 10 KB, 50 KB, 100 KB, 500 KB, 1 MB, 2 MB, 5 MB)
2. Chunk the text with `SentenceAwareChunker` (default config)
3. Run `SentenceTransformerGenerator.generate(chunks)` N times per size (default: 5)
4. Measure wall time with `time.perf_counter()`; `gc.collect()` between runs
5. Report mean/min/max throughput in chunks/s and approximate LLM tokens/s ingested

## Technology

- **Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers, local inference)
- **Chunker:** `SentenceAwareChunker` (raglet built-in)
- **Timing:** `time.perf_counter()` (high-resolution wall clock)
- **No external services** — everything runs locally

## Usage

```bash
uv run python benchmarks/embedding-throughput/run.py
uv run python benchmarks/embedding-throughput/run.py --runs 10
```

## Results

Last run on Apple M1 Pro (CPU, batch_size=256):

| Size | ~LLM tokens | Chunks | Embed time | Chunks/s | LLM tok/s ingested |
|-----:|------------:|-------:|-----------:|---------:|-------------------:|
| 5 KB | 1,280 | 3 | 0.030s | 133 | 58,324 |
| 10 KB | 2,560 | 6 | 0.037s | 167 | 74,440 |
| 50 KB | 12,800 | 28 | 0.145s | 196 | 98,776 |
| 100 KB | 25,600 | 56 | 0.275s | 205 | 103,395 |
| 500 KB | 128,000 | 280 | 1.385s | 203 | 102,749 |
| 1 MB | 262,144 | 574 | 2.776s | 207 | 104,602 |
| 2 MB | 524,288 | 1,148 | 5.551s | 207 | 104,629 |
| 5 MB | 1,310,720 | 2,870 | 14.026s | 205 | 103,566 |

Throughput stabilises at ~205 chunks/s (~100k LLM tokens/s) for texts above 50 KB.

Raw results: [`results.json`](results.json)
