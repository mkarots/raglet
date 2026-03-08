# Decision 008: Instance Lifecycle Management

**Date:** March 2026  
**Status:** Accepted

## Context

`RAGlet` instances own resources that Python's GC cannot reliably clean up: native FAISS C++ memory, `loky` worker pools from `sentence-transformers`, and SQLite file handles. Without explicit cleanup, these accumulate across instances and cause segfaults in multi-instance workflows (test loops, batch pipelines, servers).

The fix requires a cleanup contract, but the design question is: what cleanup API do we expose, and what do we require of users?

## Decision

Add `close()` to `RAGlet` and all owned components. Back it with `__del__` as a silent safety net. Optionally expose `__enter__`/`__exit__` as a convenience — but never require it.

Users have three valid patterns, all supported:

```python
# Pattern 1: implicit — GC cleans up via __del__, timing not guaranteed
raglet = RAGlet.load(...)

# Pattern 2: explicit — deterministic, recommended for loops and pipelines
raglet = RAGlet.load(...)
raglet.close()

# Pattern 3: context manager — guaranteed cleanup, idiomatic for scoped use
with RAGlet.load(...) as raglet:
    ...
```

## Rationale

- Forcing the context manager pattern is too opinionated — it breaks simple scripting use cases and surprises users coming from a "just load and query" mental model
- `__del__` alone is insufficient — CPython does not guarantee when it runs, and circular references can prevent it entirely
- `close()` as the canonical method matches the pattern used by file handles, DB connections, and other resource-owning objects in the Python stdlib
- `__enter__`/`__exit__` adds zero cost and makes the right thing easy for users who want it

## Consequences

- `FAISSVectorStore.close()` calls `index.reset()` then deletes the index — frees C++ heap immediately
- `SentenceTransformerGenerator.close()` shuts down the `loky` executor with `wait=True` — eliminates semaphore leaks
- `SQLiteStorageBackend.close()` closes the connection — releases file locks
- `RAGlet.close()` cascades to all owned components in order
- `__del__` calls `close()` on all components as a best-effort fallback
- No breaking changes — existing code with no cleanup continues to work, just with non-deterministic GC timing
- Test suite should add explicit `raglet.close()` between iterations (or use context managers) to get deterministic behavior

## Components Affected

1. `FAISSVectorStore` — `close()` resets and deletes the FAISS index
2. `SentenceTransformerGenerator` — `close()` shuts down loky worker pool
3. `SQLiteStorageBackend` — `close()` closes the SQLite connection
4. `RAGlet` — `close()` cascades, `__del__` backs it up, `__enter__`/`__exit__` wraps it
