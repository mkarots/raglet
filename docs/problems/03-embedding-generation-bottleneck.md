# Problem 03: Embedding Generation CPU Bottleneck at Scale

**Date:** 2026-03-07
**Severity:** Critical
**Component:** `raglet/embeddings/generator.py` — `SentenceTransformerGenerator.generate()`
**Related:** [Bottlenecks at Scale](../BOTTLENECKS_AT_SCALE.md), [Problem 02](02-vstack-quadratic-copy.md)

---

## 1. The Problem

Embedding generation is the single most time-consuming operation in the raglet pipeline. For 1 million chunks, it dominates total wall-clock time by an order of magnitude over every other step combined.

The `generate()` method passes all chunk texts to `sentence-transformers`, which runs a neural network forward pass on every chunk:

```python
# generator.py:169-175
batch_embeddings = self.model.encode(
    texts,
    normalize_embeddings=False,
    show_progress_bar=False,
    convert_to_numpy=True,
    batch_size=batch_size,  # 1000
)
```

The model (`all-MiniLM-L6-v2` by default) is a 22M-parameter transformer. On CPU, it encodes roughly 1,000–3,000 sentences per second depending on text length and hardware. The batch size of 1000 is passed to `sentence-transformers` as an internal batching hint, but the throughput ceiling is the model's forward pass speed.

---

## 2. Concrete Numbers

### 2.1 Time Estimates

| Chunks | CPU (~2,000 sents/s) | GPU (~50,000 sents/s) |
|--------|----------------------|------------------------|
| 10,000 | ~5 seconds | < 1 second |
| 100,000 | ~50 seconds | ~2 seconds |
| 500,000 | ~4 minutes | ~10 seconds |
| 1,000,000 | ~8 minutes | ~20 seconds |

These are rough estimates for `all-MiniLM-L6-v2` with average chunk lengths of ~200 characters. Longer chunks are slower; shorter chunks are faster. Real throughput varies by CPU generation, RAM bandwidth, and thermal throttling.

### 2.2 Memory During Embedding

The `generate()` method pre-allocates the output array upfront:

```python
# generator.py:162
output = np.empty((total_chunks, self._dimension), dtype=np.float32)
```

At 1M chunks × 384 dims × 4 bytes = **1.5 GB** allocated before the first batch encodes. This is on top of the model itself (~100–400 MB depending on framework overhead).

The `model.encode()` call also holds intermediate tensors (activations, attention matrices) during each batch. For batch_size=1000 these are small (~tens of MB), but they add to the footprint.

### 2.3 Where Time Goes in the Full Pipeline

For a typical `from_files()` call with 100k chunks:

| Step | Approximate time | % of total |
|------|-----------------|------------|
| File extraction + chunking | 1–3 seconds | ~3% |
| **Embedding generation** | **~50 seconds** | **~90%** |
| FAISS `normalize_L2` + `index.add` | 1–2 seconds | ~2% |
| `np.vstack` (embedding cache) | < 1 second | ~1% |
| Save to storage | 1–5 seconds | ~4% |

Embedding generation dominates. Everything else is noise by comparison.

---

## 3. Why It's Hard to Fix

### 3.1 The Model Is the Bottleneck, Not the Code

The `generate()` implementation is already efficient:
- Pre-allocates the output array (no vstack)
- Extracts all texts in a single pass
- Delegates batching to `sentence-transformers` (which handles internal padding and batching)
- Uses `convert_to_numpy=True` to avoid extra tensor-to-numpy conversions

There is no easy algorithmic speedup. The cost is in running the neural network forward pass on every chunk.

### 3.2 CPU Is the Default

`EmbeddingConfig.device` defaults to auto-detection via `_select_device()`:

```python
# config.py:7-24
def _select_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except ImportError:
        pass
    return "cpu"
```

This auto-detects CUDA and MPS, but most developer machines (especially CI) run on CPU. The 10–50× GPU speedup is only available when hardware is present.

---

## 4. Current Gaps

### 4.1 No Progress Reporting During Embedding

The `generate()` method sets `show_progress_bar=False`. The only progress messages are printed by the caller (`from_files()`) before and after the embedding call:

```python
# rag.py:233
output.progress("Generating embeddings...")

# ...entire embedding generation happens here, silently...

# rag.py:245
output.progress("Indexing vectors...")
```

For 1M chunks on CPU (~8 minutes), the user sees "Generating embeddings..." and then nothing for 8 minutes. No batch progress, no ETA, no way to know if the process is stuck or working.

### 4.2 No Checkpointing or Resume

If embedding generation crashes or is interrupted at batch 800 of 1000:
- All 800 batches of work are lost
- The entire process must restart from batch 0
- At 8 minutes per run, this is extremely painful

### 4.3 No Incremental Embedding

The pipeline always re-embeds everything. If you add 100 new files to an existing corpus of 100k files, the current code re-chunks everything and re-embeds from scratch (via `from_files()`). The `add_files()` method is incremental, but it still calls `add_chunks()` which calls `generate()` on the new chunks synchronously with no batching control.

### 4.4 Batch Size Mismatch

`EmbeddingConfig.batch_size` is 32, but `generate()` hardcodes `batch_size = 1000`:

```python
# config.py:84
batch_size: int = 32

# generator.py:160
batch_size = 1000
```

The config value is ignored. This isn't necessarily wrong (sentence-transformers' internal batching at 1000 is efficient), but it means the config field is misleading.

---

## 5. Solution Options

### Option A: Progress Reporting — Low effort, high UX value

Add batch-level progress reporting to `generate()`:

```python
def generate(self, chunks: list[Chunk], callback=None) -> np.ndarray:
    batch_size = 1000
    texts = [chunk.text for chunk in chunks]
    output = np.empty((len(chunks), self._dimension), dtype=np.float32)

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_result = self.model.encode(
            texts[start:end],
            normalize_embeddings=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        output[start:end] = batch_result

        if callback:
            callback(end, len(texts))

    return output
```

**Trade-off:** Adds a loop instead of a single `encode()` call, but the overhead is negligible compared to model inference. Progress callback follows the existing `CLIOutput` pattern.

**Alternative:** Use `show_progress_bar=True` for CLI contexts. Sentence-transformers has built-in tqdm support.

### Option B: Checkpointing — Medium effort, high resilience

Write embeddings to a temporary file after each batch. On restart, detect the checkpoint and resume from where it left off:

```python
def generate_with_checkpoint(self, chunks, checkpoint_path=None):
    if checkpoint_path and Path(checkpoint_path).exists():
        partial = np.load(checkpoint_path)
        start_idx = len(partial)
    else:
        partial = np.empty((0, self._dimension), dtype=np.float32)
        start_idx = 0

    remaining_texts = [c.text for c in chunks[start_idx:]]
    # ... encode remaining, append to partial, save checkpoint ...
```

**Trade-off:** Adds file I/O between batches and a checkpoint path parameter. Useful for long-running batch jobs; unnecessary for small corpora.

### Option C: Streaming Pipeline — Medium effort, structural change

Instead of generating all embeddings in one call, process the pipeline in streaming batches:

1. Read + chunk a batch of files
2. Embed the batch
3. Add to FAISS
4. Discard the batch from memory
5. Repeat

This caps memory at batch size and naturally provides progress boundaries.

**Trade-off:** Requires restructuring `from_files()` and `__init__`. The current design passes all chunks to `__init__` at once. A streaming approach would build the RAGlet incrementally.

### Option D: Respect Config Batch Size — Low effort, correctness fix

Use `self.config.batch_size` instead of hardcoding 1000:

```python
batch_size = self.config.batch_size  # from EmbeddingConfig
```

**Trade-off:** The config default of 32 would be slower than 1000 for large datasets. Consider raising the default or documenting that this controls model-level batching (GPU memory trade-off).

### Option E: GPU Auto-Detection Is Already Done — Document It

`_select_device()` already auto-detects CUDA and MPS. The remaining gap is documentation and user awareness. Many users don't know they can get 10–50× speedup by having PyTorch with CUDA installed.

### Option F: Model Selection Guidance — Documentation

Different models have dramatically different throughput:

| Model | Dimensions | Params | Relative speed |
|-------|-----------|--------|----------------|
| `all-MiniLM-L6-v2` (default) | 384 | 22M | 1× (baseline) |
| `all-MiniLM-L12-v2` | 384 | 33M | ~0.6× |
| `all-mpnet-base-v2` | 768 | 109M | ~0.3× |
| `paraphrase-MiniLM-L3-v2` | 384 | 17M | ~1.3× |

For very large corpora on CPU, a smaller model or dimensionality-reduced model can make a meaningful difference.

---

## 6. Recommendation

| Priority | Option | Impact | Effort |
|----------|--------|--------|--------|
| **1** | **A: Progress reporting** | UX — users know what's happening | Low |
| **2** | **D: Respect config batch size** | Correctness — config does what it says | Low |
| **3** | **B: Checkpointing** | Resilience — no lost work on crash | Medium |
| **4** | **C: Streaming pipeline** | Memory + UX — caps footprint, natural progress | Medium–High |
| **5** | **E + F: Documentation** | Awareness — GPU speedup, model choices | Low |

Options A and D can be done immediately with minimal risk. Option C is a structural improvement that overlaps with the deferred materialization work in ADR-009.

---

## 7. Files to Change

| File | Change |
|------|--------|
| `raglet/embeddings/generator.py` | Add batch loop with callback (Option A); use config batch size (Option D) |
| `raglet/embeddings/interfaces.py` | Add optional `callback` parameter to `generate()` signature |
| `raglet/core/rag.py` | Pass progress callback from `__init__` / `from_files()` to `generate()` |
| `raglet/config/config.py` | Consider raising default `batch_size` or documenting its meaning |
| Documentation | GPU setup guide, model selection guide |

---

## 8. Verification

- **Progress reporting:** Run `raglet create data/50mb_text -o test.raglet` and confirm batch progress is visible during embedding
- **Config batch size:** Set `batch_size=64` in config, verify `generate()` uses it
- **Checkpointing:** Start a large embedding job, kill it mid-way, restart and confirm it resumes
- **Performance:** Compare wall-clock time at 10k chunks before and after changes to confirm no regression
