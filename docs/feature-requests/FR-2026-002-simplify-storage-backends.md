# Feature Request: Clarify Storage Backend Roles — Directory as Default, SQLite for Agent Memory

| Field | Value |
|---|---|
| **FR-ID** | FR-2026-002 |
| **Component** | `raglet.storage` |
| **Priority** | Medium |
| **Type** | Simplification / Architecture |
| **Date** | 2026-03-08 |
| **Status** | Proposed |
| **Related** | ADR-005 |

---

## Summary

Make `DirectoryStorageBackend` the default. Retain `SQLiteStorageBackend` for the agent memory use case, where true incremental appends matter. Fix the directory backend's incremental weakness by switching to JSON Lines. Remove or demote `ZipStorageBackend` to a convenience export only.

---

## Problem Statement

raglet ships three storage backends with no clear guidance on when to use each. The current default is SQLite, which is the right choice for agent memory loops but the wrong default for the common case of "index a codebase and search it."

The two primary use cases have genuinely different storage needs:

**Development / static corpora** — build once, search many times, git-track the result. A directory of plain files is ideal: human-readable, inspectable without tooling, diffs cleanly in git.

**Agent memory loops** — a process that appends conversation turns or observations incrementally over time. SQLite is genuinely better here: true SQL INSERTs vs. the directory backend rewriting `embeddings.npy` on every append.

The current setup conflates these by making SQLite the default for everything, and maintains three backends without clearly documenting what each is for.

---

## Proposed Solution

1. Make `DirectoryStorageBackend` the default backend
2. Switch `chunks.json` to `chunks.jsonl` (JSON Lines) for truly incremental chunk appends in the directory backend
3. Retain `SQLiteStorageBackend` explicitly for the agent memory use case
4. Demote `ZipStorageBackend` to a convenience export (read-only, no incremental updates)
5. Document the choice clearly so users pick the right backend for their use case

The resulting storage story:

| Format | Use when |
|--------|----------|
| `.raglet/` directory | Default — development, static corpora, git-tracked knowledge bases |
| `.sqlite` | Agent memory loops — frequent incremental appends, single-file deployment |
| `.zip` | Export/archive only — sharing, no further updates |

---

## Expected Impact

- Directory becomes the obvious default for the common case
- SQLite retains a clear, justified role rather than being "the default for everything"
- Directory incremental appends become truly incremental for chunks (JSONL)
- Documentation accurately reflects when to use each format

---

## Acceptance Criteria

- `DirectoryStorageBackend` is the default when no backend is specified
- `chunks.jsonl` replaces `chunks.json` with append-only writes for incremental updates
- `SQLiteStorageBackend` is documented as the recommended backend for agent memory use cases
- `ZipStorageBackend` is documented as export-only with no incremental support
- Usage patterns documentation updated to reflect the backend guidance
- All existing tests pass

---

## Risks

- **Breaking change** for users already relying on SQLite as the default. Mitigated by auto-detecting format on `RAGlet.load()` (already implemented) and clear migration notes.
- `.npy` embeddings in the directory backend still require a full rewrite on incremental update. Acceptable at workspace scale — a 5 MB codebase saves in ~21ms. Worth documenting explicitly.
