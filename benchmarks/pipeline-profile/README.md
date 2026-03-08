# Pipeline Stage Profiler

## Purpose

Profiles each stage of the raglet build pipeline **in isolation**, showing exactly where time is spent when converting files into a searchable index. Useful for identifying bottlenecks when processing your own data.

## Method

Measures six stages independently using `time.perf_counter()`:

1. **File discovery** — `expand_file_inputs()` to resolve paths/globs
2. **Text extraction** — per-file I/O through `DocumentExtractor`
3. **Chunking** — `SentenceAwareChunker` splitting
4. **Embedding generation** — `SentenceTransformerGenerator.generate()`
5. **FAISS indexing** — normalize + add vectors
6. **Save** — serialize to directory or SQLite

Also tracks peak RSS via `resource.getrusage()` (macOS/Linux).

## Technology

- **Profiling:** `time.perf_counter()` per stage, `resource.getrusage()` for RSS
- **Input:** any file path, directory, or glob pattern
- **Save format:** directory (default) or SQLite

## Usage

```bash
uv run python benchmarks/pipeline-profile/run.py path/to/your/docs/
uv run python benchmarks/pipeline-profile/run.py docs/ --save-format sqlite
```

## Results

Output is printed to console (no JSON file). Example output format:

```
Stage            Time         %
-----------------------------------
Discovery         0.2 ms    0.0%
Extraction        3.1 ms    0.1%
Chunking          1.5 ms    0.0%
Model load      812.3 ms   22.4%
Embedding      2776.0 ms   76.5%
FAISS             8.2 ms    0.2%
Save             28.1 ms    0.8%
-----------------------------------
TOTAL          3629.4 ms  100.0%
```

Embedding generation dominates. Model load is a one-time cost (amortized over multiple builds in long-running processes).
