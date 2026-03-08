# Decision 009: Eliminate Redundant `self.embeddings` In-Memory Cache

**Date:** 2026-03-07  
**Status:** Accepted  
**Related:** FR-2026-001

---

## Context

`RAGlet` maintains `self.embeddings`, a numpy ndarray of all embedding vectors on the Python heap. The same vectors are also stored in the FAISS index (`self.vector_store`) in C++ heap memory. This duplication causes OOM crashes at scale.

### Where `self.embeddings` Is Currently Used

| Location | Usage |
|---|---|
| `RAGlet.__init__()` | Stores provided or generated embeddings; passes to `vector_store.add_vectors()` |
| `RAGlet.search()` | `.shape[1]` access for dimension validation |
| `RAGlet.add_chunks()` | `np.vstack()` append — **primary memory issue** (3× peak allocation) |
| `RAGlet.close()` | Cleared during cleanup |
| `sqlite_backend.py` L85, L138 | `raglet.embeddings[i]` and `raglet.embeddings[chunk_index]` — per-row BLOB serialisation |
| `directory_backend.py` L81–98, L155 | Saves full array as `embeddings.npy`; reads `.shape[1]` for metadata; slices for incremental saves |
| `zip_backend.py` L74, L84 | Saves full array to zip; reads `.shape[1]` for metadata |
| Tests | Manual assignment before `save()`; `.shape` validation assertions |
| `performance_test.py` | Debug logging via `.shape` |

### Why Simple Removal Is Not Possible

Storage backends require the raw embedding matrix to persist it to disk. FAISS does not currently expose a method to retrieve all indexed vectors. Removing `self.embeddings` without an alternative retrieval path would silently break serialisation.

The constraint set is:

1. FAISS does not provide a `get_all_vectors()` API in the current `VectorStore` interface
2. Embeddings cannot be regenerated at save time — inefficient and loses original values
3. Storage backends have no other source for the embedding matrix

---

## Decision

Keep `self.embeddings` as a **write-once, read-for-serialisation cache**, but eliminate the `np.vstack()` growth pattern that causes the OOM problem. The approach has two parts:

**Part 1 — Fix the growth pattern (immediate, low risk)**

Replace the unbounded vstack accumulation in `add_chunks()` with a list-of-arrays approach, deferring materialisation to serialisation time only:

```python
# Current — allocates 3× peak on every add_chunks() call:
self.embeddings = np.vstack([self.embeddings, new_embeddings])

# Replacement — O(1) append, no copy:
self._embedding_chunks.append(new_embeddings)

# Materialise only when needed (save or explicit access):
@property
def embeddings(self) -> np.ndarray:
    if len(self._embedding_chunks) > 1:
        self._embedding_chunks = [np.vstack(self._embedding_chunks)]
    return self._embedding_chunks[0] if self._embedding_chunks else self._empty_embeddings
```

This eliminates the 3× peak at `add_chunks()` time while keeping the public `.embeddings` attribute fully backward compatible for storage backends and tests.

**Part 2 — Add `get_all_vectors()` to VectorStore interface (follow-up)**

Extend the `VectorStore` interface:

```python
class VectorStore:
    def get_all_vectors(self) -> np.ndarray: ...
```

Implement for `FAISSVectorStore` using `index.reconstruct_n(0, index.ntotal)`. Once available, storage backends read directly from FAISS, and `self.embeddings` can be removed entirely in a subsequent ADR.

---

## Alternatives Considered

### 1. Remove `self.embeddings` immediately

Breaks all three storage backends and any external callers. FAISS does not currently expose a retrieval API. **Rejected** as unsafe without Part 2 in place.

### 2. Pass embeddings as parameter to `save()`

```python
raglet.save(file_path, embeddings=self.embeddings)
```

Changes the public interface of `save()` and all storage backends. Shifts the memory problem to the caller. **Rejected.**

### 3. Keep `self.embeddings` as-is, use `mmap_mode`

Only helps for on-disk loading, not the in-memory vstack. **Rejected** as a non-solution.

### 4. Temporary cache — materialise only during save, clear after

```python
def save(self, ...):
    self._materialise_embeddings()
    backend.save(...)
    del self.embeddings
```

Fragile lifecycle, subtle bugs if `save()` is not called. **Rejected.**

### 5. Deferred materialisation via list-of-arrays ✓

Zero-copy appends during `add_chunks()`, single materialisation on first access. Backward compatible. Solves the immediate OOM. **Accepted for Part 1.**

---

## Consequences

**Positive**
- Eliminates the 3× peak allocation during `add_chunks()` — resolves the OOM crash
- ~30–45 MiB reduction in peak memory per `add_chunks()` call at 10k chunks
- Fully backward compatible — storage backends, tests, external callers unaffected
- Lays groundwork for full removal once `get_all_vectors()` is implemented

**Negative**
- `self.embeddings` remains in the object — memory not fully reclaimed until Part 2
- First `save()` or `.embeddings` access after many `add_chunks()` calls incurs a single unavoidable vstack (one-time cost, not repeated)
- Part 2 requires non-trivial FAISS interface work

**Neutral**
- Public API unchanged
- No change to serialisation format or storage backends

---

## Implementation Plan

### Part 1 (this PR)

1. Add `self._embedding_chunks: list[np.ndarray]` in `__init__`
2. Replace `self.embeddings = ...` assignment in `__init__` with population of `_embedding_chunks`
3. Replace `np.vstack` in `add_chunks()` with `self._embedding_chunks.append(new_embeddings)`
4. Add `embeddings` property with lazy materialisation
5. Update `close()` to clear `_embedding_chunks`
6. Confirm all storage backend paths pass via existing tests

### Part 2 (follow-up)

1. Add `get_all_vectors() -> np.ndarray` to `VectorStore` interface
2. Implement using `index.reconstruct_n(0, index.ntotal)` in `FAISSVectorStore`
3. Migrate storage backends to call `self.vector_store.get_all_vectors()`
4. Remove `self.embeddings` / `self._embedding_chunks` entirely
5. Update tests

---

## Verification

**Part 1:** Run `performance_test.py --sizes 10000`. Expected: no OOM crash, peak memory reduced by ≥30 MiB vs baseline. All existing serialisation tests pass.

**Part 2:** Same performance test plus a round-trip save/load test confirming embeddings retrieved from FAISS match the originals to float32 precision.
