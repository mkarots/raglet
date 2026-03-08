# Feature Request: Remove Redundant `self.embeddings` Cache

| Field | Value |
|---|---|
| **FR-ID** | FR-2026-001 |
| **Component** | `raglet.core.RAGlet` |
| **Priority** | High |
| **Type** | Performance / Memory |
| **Date** | 2026-03-07 |
| **Status** | Accepted |
| **Related** | ADR-009 |

---

## Summary

Remove the `self.embeddings` numpy array from the `RAGlet` class. This field is a redundant copy of data already stored in the FAISS vector store, and causes unnecessary memory consumption and peak allocation spikes.

---

## Problem Statement

`RAGlet` currently maintains two copies of all embedding vectors in memory simultaneously:

- `self.embeddings` — a numpy ndarray on the Python heap
- `self.vector_store` (FAISSVectorStore) — the FAISS index, which already holds the same vectors in C++ heap memory

This duplication is the primary contributor to memory exhaustion during performance testing at scale. It manifests as:

- Peak memory during `np.vstack()` reaching 2–3× the steady-state size
- Monotonically growing memory footprint across repeated `add_chunks()` calls
- OOM crashes when approaching the 400 MiB memory ceiling

Memory profiling (`performance_test.py --sizes 10000`) shows the process hitting exactly 400 MiB at crash time, with a characteristic steady-climb pattern in the 14–26 second window corresponding directly to repeated vstack allocations.

---

## Root Cause

In `add_chunks()`, the following line accumulates memory on every call:

```python
self.embeddings = np.vstack([self.embeddings, new_embeddings])
```

For N chunks at dimension D (default 384), each vstack allocates a new output array while holding both old and new arrays simultaneously — a peak of ~3× the final array size.

A full codebase grep confirms `self.embeddings` is used in exactly **one place** outside of assignment: the `.shape[1]` dimension check in `search()`, which can be obtained directly from the embedding generator.

---

## Proposed Solution

Replace the dimension check in `search()`:

```python
# Before
stored_dim = self.embeddings.shape[1]

# After
stored_dim = self.embedding_generator.get_dimension()
```

Then remove all assignments to `self.embeddings` in `__init__`, `add_chunks()`, and `close()`.

---

## Expected Impact

For 10k chunks at 384-dim float32:

- **Steady-state**: ~15 MiB permanently freed
- **Peak during add_chunks**: ~30–45 MiB freed
- **Total effective reduction**: up to ~60 MiB per `add_chunks` call at scale

---

## Acceptance Criteria

- `self.embeddings` removed from `__init__`, `add_chunks()`, and `close()`
- `search()` dimension validation uses `self.embedding_generator.get_dimension()`
- All existing tests pass
- `performance_test.py --sizes 10000` completes without OOM crash
- Peak memory during `add_chunks` reduced by at least 30 MiB vs baseline

---

## Risks

Low risk. The only behavioral change is removal of a redundant field. Before merging, grep the full codebase for `.embeddings` references to confirm no external callers depend on the attribute.
