<div align="center">
  <img src="assets/logo.png" alt="raglet logo" width="600">
</div>

# raglet

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Portable memory for small text corpora. No servers, no API keys, no infrastructure.**

There's a class of knowledge that's too big for a prompt but too small to justify a vector database: a codebase, a Slack export, a folder of meeting notes. raglet turns that text into a searchable directory you can save, git commit, or carry to another machine.

```bash
pip install raglet
```

---

## How it works

```python
from raglet import RAGlet

# Build a searchable index from your files
rag = RAGlet.from_files(["docs/", "notes.md"])

# Search semantically
results = rag.search("what did we decide about the API design?", top_k=5)

for chunk in results:
    print(f"[{chunk.score:.2f}] {chunk.source}")
    print(chunk.text)
    print()

# Save to a portable directory
rag.save(".raglet/")
```

**Example output:**
```
[0.87] docs/decisions/api-design.md
We decided to keep the API surface minimal — just search(), add_text(), and save().
The goal is that a new user can be productive in under 5 minutes.

[0.81] notes/2024-03-meeting.md
API design discussion: favour explicit save() calls over auto-persistence.
Incremental updates should be opt-in, not default behaviour.

[0.74] docs/decisions/api-design.md
The search() method returns ranked chunks with scores. The caller decides
what to do with them — raglet does not call any LLM.
```

Load it back anywhere:

```python
rag = RAGlet.load(".raglet/")
results = rag.search("your query")
```

---

## When to use raglet

raglet is designed for workspace-scale corpora. The embedding pipeline processes ~95K LLM tokens/sec on Apple Silicon (MPS). Build is a one-time cost — after that, search stays under 15 ms regardless of dataset size.

| Corpus size | LLM tokens | Build time (MPS) | Search p50 | raglet? |
|-------------|------------|------------------|------------|---------|
| < 8 KB | < 10K | — | — | Use a prompt directly |
| 8 KB – 1 MB | 10K – 262K | ~3.5s | ~4 ms | ✅ Sweet spot |
| 1 – 10 MB | 262K – 2.6M | 3.5–35s | ~6 ms | ✅ Works great |
| 10 – 100 MB | 2.6M – 26M | 35s – 6 min | ~10 ms | ⚠️ Works — build is a one-time cost |
| > 100 MB | > 26M | hours | — | ❌ Use a vector database instead |

If your corpus is larger than ~100 MB, raglet is the wrong tool. Use a persistent vector database (Chroma, Weaviate, Pinecone) instead.

---

## The `.raglet/` directory

When you save a raglet, you get a plain, inspectable directory:

```
.raglet/
├── config.json      # chunking, embedding model, search settings
├── chunks.json      # all text chunks with source and metadata
├── embeddings.npy   # NumPy float32 embeddings matrix
└── metadata.json    # version, timestamps, chunk count, dimensions
```

Everything is human-readable JSON (except the embeddings binary). That means you can:

```bash
# Inspect your chunks
cat .raglet/chunks.json

# Check what model and config was used
cat .raglet/config.json

# Git commit the whole thing
git add .raglet/ && git commit -m "update knowledge base"

# Export for sharing
raglet package --raglet .raglet/ --format zip --out knowledge.zip
```

No proprietary format. No lock-in. Your data is always accessible.

---

## Installation

```bash
pip install raglet
```

Or with Docker — no install needed:

```bash
docker pull mkarots/raglet
docker run -v /path/to/project:/workspace mkarots/raglet build docs/ --out .raglet/
```

> **Note:** Alpine Linux is not supported. Use `python:3.11-slim` or similar images.

---

## CLI

```bash
# Build a knowledge base
raglet build docs/ --out .raglet/
raglet build docs/ src/ "*.md" --out .raglet/ --chunk-size 1024

# Search it
raglet query "how does authentication work?" --raglet .raglet/
raglet query "what is X?" --raglet memory.sqlite --top-k 10

# Add files, directories, or glob patterns incrementally
raglet add new_file.txt --raglet .raglet/
raglet add new-docs/ --raglet .raglet/
raglet add "*.md" --raglet .raglet/ --ignore __pycache__

# Convert between formats
raglet package --raglet .raglet/ --format zip --out export.zip
raglet package --raglet .raglet/ --format sqlite --out memory.sqlite
```

---

## Storage formats

raglet supports three formats. All load with `RAGlet.load()` — format is auto-detected from the path.

| Format | Use when | Incremental updates |
|--------|----------|---------------------|
| `.raglet/` directory | Default — development, git-tracked knowledge bases | ✅ |
| `.sqlite` | Agent memory loops — frequent appends, single-file deployment | ✅ True appends |
| `.zip` | Export and sharing only | ❌ Read-only |

```python
rag.save(".raglet/")          # directory (default)
rag.save("memory.sqlite")     # SQLite — true incremental appends
rag.save("export.zip")        # zip archive

rag = RAGlet.load(".raglet/")
rag = RAGlet.load("memory.sqlite")
rag = RAGlet.load("export.zip")
```

**When to use SQLite:** if you're running an agent loop that appends conversation turns or observations continuously, SQLite is the better choice — it does true SQL `INSERT` operations rather than rewriting files on each save.

---

## Common patterns

### Load or create

```python
from pathlib import Path
from raglet import RAGlet

rag = RAGlet.load(".raglet/") if Path(".raglet/").exists() else RAGlet.from_files(["docs/"])
```

### Use with any LLM

```python
results = rag.search("user query", top_k=5)
context = "\n\n".join(chunk.text for chunk in results)

# Pass context to your LLM of choice
response = your_llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
```

raglet handles retrieval. You handle generation.

### Agent loop with persistent memory

```python
from pathlib import Path
from raglet import RAGlet

# SQLite is the right format for agent memory — true incremental appends
path = "memory.sqlite"
rag = RAGlet.load(path) if Path(path).exists() else RAGlet.from_files([])

while True:
    query = input("You: ")
    if query == "exit":
        rag.save(path)
        break

    results = rag.search(query, top_k=5)
    response = your_llm(results, query)

    rag.add_text(query, source="user")
    rag.add_text(response, source="assistant")
    rag.save(path, incremental=True)
```

### Incremental updates (cheap appends)

The initial `from_files()` is the expensive step — it embeds all the text. After that, appending new content only embeds the **new** chunks. A 100 KB file appends in ~0.3s regardless of how large the existing raglet is.

```python
# Add files, directories, or glob patterns
rag.add_file("new_doc.txt")
rag.add_files(["file1.txt", "file2.md"])
rag.add_files(["new-docs/"])

# Add raw text
rag.add_text("Some text", source="manual")

# Save incrementally (only writes new data)
rag.save(".raglet/", incremental=True)
```

See [Usage Patterns](docs/USAGE_PATTERNS.md) for the full build-once-append-search workflow.

---

## Configuration

```python
from raglet import RAGlet, RAGletConfig

config = RAGletConfig()
config.chunking.size = 1024
config.chunking.overlap = 100
config.embedding.model = "all-mpnet-base-v2"

rag = RAGlet.from_files(["docs/"], config=config)
```

Available embedding models: `all-MiniLM-L6-v2` (default, fast), `all-mpnet-base-v2` (higher quality), `BAAI/bge-small-en-v1.5`.

Search with a similarity threshold:

```python
results = rag.search("query", top_k=10, similarity_threshold=0.7)
```

---

## Known limitations

**File formats:** v0.1.0 supports `.txt` and `.md` files only. PDF, DOCX, and HTML are on the roadmap. For unsupported formats, extract text first and use `add_text()`.

**Corpus size:** raglet is workspace-scale, not internet-scale. Search stays under 15 ms up to 100 MB, but build time becomes significant (100 MB takes ~6 minutes on MPS). Above ~100 MB, use a proper vector database.

**No file change detection:** raglet does not watch for file changes. If a file is modified, rebuild from scratch with `from_files()`. Incremental updates (`add_file`, `add_files`) are for adding new files only.

**CPU-only machines:** embedding is ~10–20× slower without a GPU or MPS. Search latency (<10 ms) is hardware-independent and unaffected.

---

## Features

- ✅ Text extraction from `.txt` and `.md` files
- ✅ Sentence-aware chunking
- ✅ Local embeddings via sentence-transformers (no API keys)
- ✅ Vector search via FAISS
- ✅ Three portable formats: directory, SQLite, zip
- ✅ Incremental updates
- ✅ CLI — `build`, `query`, `add` (files, directories, globs), `package`
- ✅ Docker image

---

## Principles

**Portable** — One directory (or file). Git commit it, email it, load it on another machine.

**Small by design** — Workspace-scale: codebases, conversations, notes. Not the internet.

**Retrieval only** — raglet finds chunks. You decide what to do with them. Bring your own LLM.

**Open format** — JSON files you can read, edit, and extract. No proprietary format, no lock-in.

**Zero infrastructure** — `pip install raglet` or `docker run`. That's it.

---

## Roadmap

```
v0.1.0 (now)   Semantic search, save/load, incremental updates, CLI
v0.2.0         PDF, DOCX, HTML support
v0.3.0         File change detection (rebuild only changed files)
```

---

## Development

```bash
# Install with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
make install-dev

# Run tests
make test           # all tests
make test-unit      # unit only
make test-e2e       # end-to-end only

# Code quality
make lint
make format
make type-check
make ci             # full pipeline
```

---

## Architecture

```
raglet/
├── core/           # domain models and orchestrator
├── processing/     # document extraction and chunking
├── embeddings/     # embedding generation
├── vector_store/   # vector storage and search
├── storage/        # file serialization (dir / sqlite / zip)
└── config/         # configuration system
```

See [docs/proposals/ARCHITECTURE.md](docs/proposals/ARCHITECTURE.md) for design decisions.

---

## Documentation

- [Problem Statement](docs/problems/00-problem-statement.md)
- [Architecture Decisions](docs/decisions/)
- [Usage Patterns](docs/USAGE_PATTERNS.md)
- [Benchmark Results](benchmarks/SCALE_REPORT.md)
- [Launch Plan](docs/plans/LAUNCH_PLAN_v0.1.0.md)

---

## License

MIT
