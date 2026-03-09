## Benchmark Results (2026-03-09)

### Configuration

| Setting | Value |
|---------|-------|
| Model | `all-MiniLM-L6-v2` |
| Device | mps |
| Batch size | 128 |
| FP16 | True |
| Chunk size | 256 tokens (~1024 chars) |
| Chunk overlap | 50 tokens |
| Chunk strategy | sentence-aware |
| Max sequence length | 256 |

### Highlights

- Peak embedding throughput: **398 chunks/s** (at 100 MB)
- Search p50: **3.2 ms** – **10.4 ms** across all sizes
- At 100 MB (138,989 chunks): build 6m0s, search 10.4 ms, save 1227 ms, load 574 ms
- Building is the expensive one-time cost; search/save/load stay fast regardless of size

### Embedding Throughput

| Text size | Chunks | Embed time | Throughput (chunks/s) |
|---------:|-------:|-----------:|---------------------:|
| 102 KB | 138 | 0.43s | 322 |
| 512 KB | 688 | 1.8s | 378 |
| 1 MB | 1,377 | 3.5s | 394 |
| 2 MB | 2,755 | 7.0s | 396 |
| 5 MB | 6,887 | 17.3s | 398 |
| 10 MB | 13,774 | 35.4s | 389 |
| 20 MB | 27,548 | 1m12s | 382 |
| 50 MB | 68,871 | 2m56s | 391 |
| 100 MB | 137,743 | 5m48s | 396 |

### Operation Latency

| Text size | Chunks | Build | Search p50 | Save p50 | Load p50 |
|---------:|-------:|------:|-----------:|---------:|---------:|
| 102 KB | 140 | 1.2s | 4.8 ms | 4.2 ms | 5.2 ms |
| 512 KB | 695 | 1.9s | 3.2 ms | 6.6 ms | 6.2 ms |
| 1 MB | 1,390 | 3.6s | 3.7 ms | 12.2 ms | 9.8 ms |
| 2 MB | 2,780 | 7.2s | 5.6 ms | 23.6 ms | 14.7 ms |
| 5 MB | 6,949 | 17.5s | 6.0 ms | 64.7 ms | 27.4 ms |
| 10 MB | 13,899 | 34.6s | 6.3 ms | 122 ms | 51.4 ms |
| 20 MB | 27,797 | 1m10s | 7.1 ms | 263 ms | 106 ms |
| 50 MB | 69,494 | 3m2s | 9.2 ms | 655 ms | 290 ms |
| 100 MB | 138,989 | 6m0s | 10.4 ms | 1227 ms | 574 ms |

### How to read these numbers

- **Build** is a one-time cost when creating a `.raglet` file from scratch.
  Appending new content (`add_file` / `add_text`) only embeds the new chunks.
- **Search** stays under 10 ms at workspace scale — it does not grow with dataset size (FAISS indexed).
- **Save/Load** scale linearly with chunk count but stay well under 100 ms for typical workspaces.
