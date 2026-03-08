# cProfile Analysis

## Purpose

Deep-dive profiling using Python's `cProfile` to find **function-level hotspots** in raglet operations. While the other benchmarks measure wall time, this one shows which specific functions consume CPU time — useful for optimisation work.

## Method

Profiles 9 operations independently with `cProfile.Profile()`:

1. **embed** — `SentenceTransformerGenerator.generate()`
2. **faiss_add** — `FAISSVectorStore.add_vectors()`
3. **sqlite_save** — `SQLiteStorageBackend.save()` (full write)
4. **sqlite_load** — `SQLiteStorageBackend.load()` (includes FAISS rebuild)
5. **sqlite_incr** — `SQLiteStorageBackend.add_chunks()` (incremental)
6. **dir_save** — `DirectoryStorageBackend.save()` (full write)
7. **dir_load** — `DirectoryStorageBackend.load()` (includes FAISS rebuild)
8. **dir_incr** — `DirectoryStorageBackend.save(incremental=True)`
9. **search** — `RAGlet.search()` x N queries

Results are filtered to show raglet + stdlib code by default (hides torch/transformers noise).

## Technology

- **Profiling:** `cProfile` + `pstats` (Python stdlib)
- **Output:** `.prof` files (compatible with snakeviz, pyprof2calltree, etc.)
- **Visualisation:** `snakeviz benchmarks/cprofile-analysis/profile_results/embed.prof`

## Usage

```bash
uv run python benchmarks/cprofile-analysis/run.py
uv run python benchmarks/cprofile-analysis/run.py --size 5000
uv run python benchmarks/cprofile-analysis/run.py --ops embed,sqlite_save,search
uv run python benchmarks/cprofile-analysis/run.py --no-raglet-filter  # show all functions
```

## Results

Output directory: `benchmarks/cprofile-analysis/profile_results/`

- One `.prof` file per operation (loadable in snakeviz)
- `summary.json` — machine-readable wall-time + CPU-time + top hotspots per operation
