# Usage Patterns

raglet has an asymmetric performance profile: **building is slow, searching is instant, appending is cheap.** Understanding this shapes how you should use it.

---

## Performance profile

Measured on Apple Silicon (MPS) with `all-MiniLM-L6-v2`, 10 MB of text (~5,800 chunks):

| Operation | Time | Scales with |
|-----------|------|-------------|
| `from_files()` (initial build) | ~5s (MPS) / ~27s (CPU) | Total text size |
| `search()` | 6–25 ms | Constant (FAISS indexed) |
| `save()` | 50–85 ms | Total chunks |
| `load()` | 30–40 ms | Total chunks |
| `add_text()` / `add_file()` | Proportional to new text only | New text size |

The initial build is a one-time cost. After that, every operation is fast.

---

## Pattern 1: Build once, search forever

The simplest pattern. Build a raglet file from your corpus once, commit it, and search it from then on.

```python
from raglet import RAGlet

# One-time build (slow — embeds everything)
rag = RAGlet.from_files(["docs/", "src/"])
rag.save("knowledge.sqlite")
```

```python
# Every subsequent use (fast — no embedding needed)
rag = RAGlet.load("knowledge.sqlite")
results = rag.search("how does authentication work?")
```

Good for: static corpora, documentation, codebases that change infrequently.

---

## Pattern 2: Build once, grow incrementally

Start with an initial build, then append new content over time. Each append only embeds the new chunks — the existing index is untouched.

```python
from pathlib import Path
from raglet import RAGlet

# First run: build from scratch
if not Path(".raglet/").exists():
    rag = RAGlet.from_files(["docs/"])
    rag.save(".raglet/")
else:
    rag = RAGlet.load(".raglet/")

# Later: add new files (only new content is embedded)
rag.add_file("docs/new-feature.md")
rag.save(".raglet/", incremental=True)
```

The `add_file()` call on a 100 KB file takes ~0.3s regardless of whether the existing raglet has 1,000 or 50,000 chunks.

Good for: growing knowledge bases, living documentation, projects where files are added over time.

---

## Pattern 3: Conversational memory

Accumulate context during a session. Each `add_text()` embeds just that snippet.

```python
from pathlib import Path
from raglet import RAGlet

path = "memory.sqlite"
rag = RAGlet.load(path) if Path(path).exists() else RAGlet.from_files([])

while True:
    query = input("You: ")
    if query == "exit":
        rag.save(path)
        break

    results = rag.search(query, top_k=5)
    context = "\n".join(r.text for r in results)
    response = your_llm(context, query)

    # Append conversation (cheap — only embeds these two strings)
    rag.add_text(query, source="user")
    rag.add_text(response, source="assistant")
```

Good for: agent loops, chat history, session memory.

---

## Pattern 4: CLI workflow

```bash
# Initial build (one-time)
raglet build docs/ src/ --out .raglet/

# Search (instant)
raglet query "what is the retry policy?" --raglet .raglet/

# Add new files incrementally (only embeds new content)
raglet add docs/new-doc.md --raglet .raglet/

# Search again (still instant, now includes new content)
raglet query "what is the retry policy?" --raglet .raglet/
```

---

## What makes appending cheap

When you call `add_text()`, `add_file()`, or `add_chunks()`:

1. Only the **new** chunks are embedded (the expensive part)
2. New vectors are appended to the FAISS index (O(new chunks), not O(total))
3. Embeddings are stored in a list, not copied into a single array
4. `save(incremental=True)` writes only the new data to disk

The existing chunks, embeddings, and index are never re-processed.

---

## When to rebuild vs append

| Scenario | Approach |
|----------|----------|
| Adding new files | `add_file()` / `add_files()` — append |
| Logging conversations | `add_text()` — append |
| File content changed | Rebuild with `from_files()` (raglet doesn't track file changes) |
| Changed chunk size or embedding model | Rebuild (embeddings are model-specific) |
| Migrating to a different format | `RAGlet.load()` then `save()` to new path |

---

## Storage format guidance

| Format | Use when |
|--------|----------|
| `.raglet/` directory | Git-tracked, inspectable, development |
| `.sqlite` | Single-file portability, production, sharing |
| `.zip` | Export/archive (read-only, no incremental updates) |

All formats load with `RAGlet.load()` — it auto-detects from the path.
