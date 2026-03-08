# Problem 02: `np.vstack` Quadratic Copy in `add_chunks()`

**Date:** 2026-03-07
**Severity:** Critical
**Component:** `raglet/core/rag.py` — `RAGlet.add_chunks()`
**Related:** [ADR-009](../decisions/009-remove-embeddings-cache.md), [FR-2026-001](../feature-requests/FR-2026-001-remove-embeddings-cache.md), [Bottlenecks at Scale](../BOTTLENECKS_AT_SCALE.md)

---

## 1. The Problem

Every call to `add_chunks()` runs this line:

```python
self.embeddings = np.vstack([self.embeddings, new_embeddings])
```

`np.vstack` does not append in-place. It:

1. Allocates a **new** array of shape `(existing + new, embedding_dim)`
2. Copies all existing embeddings into the new array
3. Copies the new embeddings after them
4. Discards the old array

When `add_chunks()` is called repeatedly — which is the normal pattern for incremental ingestion, `add_text()`, `add_files()`, or pipeline loops — this produces **O(n²) total copy operations** over the course of ingestion.

---

## 2. Why It's Critical

### 2.1 Peak Memory: 3× the Array Size

During a single vstack, three arrays are simultaneously live:

| Array | Size |
|-------|------|
| `self.embeddings` (old) | N × D × 4 bytes |
| `new_embeddings` | M × D × 4 bytes |
| Result of `np.vstack` | (N + M) × D × 4 bytes |

Peak memory for this operation alone: **(2N + M) × D × 4 bytes**.

At dimension D = 384 and float32:

| Existing chunks (N) | Old array | New (1k batch) | vstack result | Peak during vstack |
|---------------------|-----------|-----------------|---------------|-------------------|
| 10,000 | 15 MB | 1.5 MB | 16.5 MB | ~33 MB |
| 100,000 | 150 MB | 1.5 MB | 151.5 MB | ~303 MB |
| 500,000 | 750 MB | 1.5 MB | 751.5 MB | ~1.5 GB |
| 1,000,000 | 1.5 GB | 1.5 MB | 1.5 GB | ~3.0 GB |

At 10k chunks, performance tests already hit the ~400 MiB process ceiling and OOM.

### 2.2 Cumulative Copy Cost: O(n²)

If `add_chunks()` is called K times, each adding a batch of B chunks, the total data copied is:

```
B + 2B + 3B + ... + KB = B × K(K+1)/2 = O(K²B)
```

For 1M chunks added in batches of 1000 (K = 1000):

- Each vstack copies progressively more data
- Total bytes copied: ~750 GB
- Wall-clock overhead: tens of seconds to minutes depending on system

### 2.3 The Duplication Makes It Worse

The vstack result duplicates data that FAISS already holds in its own C++ heap (see Section 5). The Python-side array exists primarily for storage backends that need a materialized numpy array for serialization. The actual search path never reads from `self.embeddings` — it goes through FAISS.

---

## 3. Where It Manifests

### 3.1 Direct: `add_chunks()` loop

Any code that calls `add_chunks()` repeatedly:

```python
for batch in chunk_batches:
    raglet.add_chunks(batch)  # vstack on every iteration
```

### 3.2 Indirect: `add_text()` and `add_files()`

Both call `add_chunks()` internally:

```python
# add_text() → chunks text → calls add_chunks()
# add_files() → extracts + chunks files → calls add_chunks()
```

### 3.3 Pipeline: `__init__` single-shot

Even a single call via `__init__` with all chunks at once:

```python
self.embeddings = self.embedding_generator.generate(chunks)  # array 1
self.vector_store.add_vectors(self.embeddings, chunks)        # FAISS copy
```

Here vstack isn't involved, but the `self.embeddings` assignment still holds a full duplicate of what FAISS stores. Peak memory is 2× the embedding matrix.

---

## 4. What Reads `self.embeddings`

Before solving this, every consumer of `self.embeddings` must be accounted for:

| Consumer | Access pattern | Notes |
|----------|---------------|-------|
| `search()` | `self.embeddings.shape[1]` | Dimension check only; replaceable with `self.embedding_generator.get_dimension()` |
| `close()` | Clears array | Cleanup |
| `directory_backend._save_full()` | `raglet.embeddings` — full array save | Needs materialized array |
| `directory_backend._add_chunks_incremental()` | `raglet.embeddings[current_count:]` — slice | Needs materialized array |
| `sqlite_backend._save_full()` | `raglet.embeddings[i]` — per-row iteration | Needs materialized array |
| `sqlite_backend._add_chunks_incremental()` | `raglet.embeddings[current_count + i]` — per-row iteration | Needs materialized array |
| `zip_backend._save_full()` | `raglet.embeddings` — full array save | Needs materialized array |
| Tests | `.shape` assertions, manual assignment | Backward compatibility |

**Key observation:** All read access happens during **serialization** (`save()`), never during the hot path (`add_chunks()`, `search()`). The hot path only writes.

---

## 5. Solution Options

### Option A: Deferred Materialization (ADR-009 Part 1) — Recommended first step

Replace the monolithic `self.embeddings` array with a list of array fragments. Materialize into a single array only when a consumer reads `.embeddings`:

```python
# In __init__:
self._embedding_chunks: list[np.ndarray] = []

# In add_chunks() — O(1) append, no copy:
self._embedding_chunks.append(new_embeddings)

# Property — materialize on first read:
@property
def embeddings(self) -> np.ndarray:
    if len(self._embedding_chunks) > 1:
        self._embedding_chunks = [np.vstack(self._embedding_chunks)]
    return self._embedding_chunks[0] if self._embedding_chunks else self._empty_embeddings
```

**Characteristics:**
- `add_chunks()` becomes O(1) per call — just a list append
- Total copy cost: one vstack when `.embeddings` is first read (typically on `save()`)
- Peak memory: same final size, but the 3× peak happens once instead of on every call
- Backward compatible: all consumers see the same `.embeddings` array
- Risk: low — property is transparent to callers

**This is accepted in ADR-009 Part 1.**

### Option B: FAISS Reconstruction (ADR-009 Part 2) — Full elimination

Add `get_all_vectors()` to the `VectorStore` interface. Implement via FAISS `index.reconstruct_n()`. Migrate storage backends to read from FAISS. Remove `self.embeddings` entirely.

```python
# VectorStore interface:
def get_all_vectors(self) -> np.ndarray: ...

# FAISSVectorStore implementation:
def get_all_vectors(self) -> np.ndarray:
    n = self.index.ntotal
    if n == 0:
        return np.empty((0, self.embedding_dim), dtype=np.float32)
    return self.index.reconstruct_n(0, n)
```

**Characteristics:**
- Removes ~1.5 GB at 1M chunks (the entire Python-side copy)
- No vstack anywhere — problem fully eliminated
- Storage backends call `vector_store.get_all_vectors()` instead of `raglet.embeddings`
- Risk: medium — requires interface change, storage backend migration, and verification that `reconstruct_n` returns vectors at float32 precision after normalization

**Important caveat:** FAISS `IndexFlatIP` stores **normalized** vectors (after `faiss.normalize_L2()`). Reconstructed vectors will be the L2-normalized form, not the original embeddings. Storage backends currently save the original (pre-normalization) embeddings via `self.embeddings`. If round-trip fidelity to original vectors matters, this needs careful handling:
- Either store normalization state and denormalize on reconstruction
- Or accept that saved embeddings are always normalized (which is fine for search, since they'll be re-normalized on load anyway)

### Option C: Pre-allocated Buffer with Fill Pointer

Allocate a large numpy array upfront and track how much is filled:

```python
# In __init__:
capacity = max(len(chunks) * 2, 1024)  # initial capacity
self._embeddings_buffer = np.empty((capacity, dim), dtype=np.float32)
self._embeddings_count = 0

# In add_chunks():
needed = self._embeddings_count + len(new_embeddings)
if needed > len(self._embeddings_buffer):
    new_capacity = max(needed, len(self._embeddings_buffer) * 2)
    new_buffer = np.empty((new_capacity, dim), dtype=np.float32)
    new_buffer[:self._embeddings_count] = self._embeddings_buffer[:self._embeddings_count]
    self._embeddings_buffer = new_buffer
self._embeddings_buffer[self._embeddings_count:needed] = new_embeddings
self._embeddings_count = needed

@property
def embeddings(self) -> np.ndarray:
    return self._embeddings_buffer[:self._embeddings_count]
```

**Characteristics:**
- Amortized O(1) append (like Python list internals)
- Geometric growth means log(n) reallocations total instead of n
- Peak memory: up to 2× the final size (due to over-allocation)
- More complex than Option A; similar memory ceiling

**Trade-off vs Option A:** Option C avoids even the one-time vstack at materialization, but wastes memory from over-allocation and adds complexity. For raglet's scale, Option A is simpler and sufficient.

### Option D: Do Nothing to `self.embeddings`, Fix the Caller

If `add_chunks()` is only called once with all chunks (not in a loop), the quadratic problem disappears. Restructure the pipeline to batch everything upfront:

```python
# Instead of:
for batch in batches:
    raglet.add_chunks(batch)

# Do:
all_chunks = [chunk for batch in batches for chunk in batch]
raglet.add_chunks(all_chunks)
```

**Characteristics:**
- No code change to RAGlet
- Only works if the caller controls the loop
- Doesn't fix `add_text()`, `add_files()`, or external callers
- Still has the 2× peak from the single vstack
- **Not a real fix** — just avoids the worst case

---

## 6. Recommendation

**Implement in order:**

1. **Option A (deferred materialization)** — Low risk, backward compatible, eliminates the O(n²) copy. This is ADR-009 Part 1 and is already accepted.

2. **Option B (FAISS reconstruction)** — Follow-up. Eliminates the Python-side embedding array entirely. Requires interface work and normalization-aware storage. This is ADR-009 Part 2.

Option C is more complex than A with marginal benefit at raglet's intended scale. Option D is not a solution.

---

## 7. Files to Change

### Part 1 (Option A)

| File | Change |
|------|--------|
| `raglet/core/rag.py` | Replace `self.embeddings` with `self._embedding_chunks` list; add `embeddings` property; update `__init__`, `add_chunks`, `close` |

No changes to storage backends, tests, or public API — the property is transparent.

### Part 2 (Option B)

| File | Change |
|------|--------|
| `raglet/vector_store/interfaces.py` | Add `get_all_vectors()` abstract method |
| `raglet/vector_store/faiss_store.py` | Implement `get_all_vectors()` via `reconstruct_n` |
| `raglet/storage/directory_backend.py` | Replace `raglet.embeddings` reads with `raglet.vector_store.get_all_vectors()` |
| `raglet/storage/sqlite_backend.py` | Replace `raglet.embeddings[i]` reads with vector store access |
| `raglet/storage/zip_backend.py` | Replace `raglet.embeddings` reads with vector store access |
| `raglet/core/rag.py` | Remove `self._embedding_chunks` and `embeddings` property |
| Tests | Update `.embeddings` assertions to use vector store |

---

## 8. Verification

**Part 1:**
- `make test` — all existing tests pass (property is transparent)
- `performance_test.py --sizes 10000` — no OOM; peak memory reduced by ≥30 MiB
- Manual: call `add_chunks()` in a loop of 100 batches; confirm constant memory during the loop (growth only on final `.embeddings` access)

**Part 2:**
- Round-trip test: create → save → load → compare embeddings to float32 precision
- `performance_test.py --sizes 10000` — peak memory reduced by ~1.5 GB equivalent
- Confirm `reconstruct_n` output matches stored embeddings (accounting for normalization)
