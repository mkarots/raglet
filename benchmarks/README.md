# Benchmarks

Performance benchmarks for raglet. Each benchmark lives in its own directory with a README explaining purpose, method, and results.

## Index

| Benchmark | What it measures | Key metric | Run command |
|-----------|-----------------|------------|-------------|
| [embedding-throughput](embedding-throughput/) | Embedding generation speed across text sizes | ~205 chunks/s on CPU | `uv run python benchmarks/embedding-throughput/run.py` |
| [latency](latency/) | p50/p95/p99 for save, load, and search | Search <10ms at 5 MB | `uv run python benchmarks/latency/run.py` |
| [pipeline-profile](pipeline-profile/) | Time breakdown per build stage | Embedding dominates (~76%) | `uv run python benchmarks/pipeline-profile/run.py <path>` |
| [storage-performance](storage-performance/) | End-to-end pipeline across formats and sizes | SQLite vs directory comparison | `uv run python benchmarks/storage-performance/run.py` |
| [measure-suite](measure-suite/) | Wall-time + RSS memory for all operations | Full operational profile | `uv run python benchmarks/measure-suite/run.py` |
| [cprofile-analysis](cprofile-analysis/) | Function-level CPU hotspots via cProfile | `.prof` files for snakeviz | `uv run python benchmarks/cprofile-analysis/run.py` |

## Quick start

Run any benchmark from the repo root:

```bash
# Fast: embedding throughput only
uv run python benchmarks/embedding-throughput/run.py

# Full latency sweep
uv run python benchmarks/latency/run.py

# Profile your own data
uv run python benchmarks/pipeline-profile/run.py path/to/docs/

# Everything (wall-time + memory)
uv run python benchmarks/measure-suite/run.py
```

## Parameter sweeps

Run multiple benchmarks with parameter grids using the sweep runner:

```bash
# Default sweep (all benchmarks, full grids)
make benchmark-sweep

# Quick sweep (smaller grids, faster feedback)
make benchmark-sweep SWEEP_CONFIG=benchmarks/sweep-quick.yaml

# Run specific benchmarks only
make benchmark-sweep SWEEP_ARGS="--only latency,embedding-throughput"

# Dry run (print commands without executing)
make benchmark-sweep SWEEP_ARGS="--dry-run"

# Direct invocation
uv run python benchmarks/sweep.py --config benchmarks/sweep.yaml
uv run python benchmarks/sweep.py --only latency --output results/latency-sweep.json
```

Sweep configs are YAML files that define a `grid` per benchmark. Every combination in the grid produces one run:

```yaml
benchmarks:
  latency:
    script: benchmarks/latency/run.py
    grid:
      runs: [10, 20]
      format: [sqlite, directory]
  # produces 4 runs: 10×sqlite, 10×directory, 20×sqlite, 20×directory
```

Configs:
- `sweep.yaml` — full sweep across all benchmarks
- `sweep-quick.yaml` — minimal parameters for fast iteration

Results are saved to `benchmarks/sweep-results.json`.

## Environment

All benchmarks run locally with no external services. Results depend on hardware — numbers in this repo were collected on Apple M1 Pro (CPU mode, `all-MiniLM-L6-v2`).
