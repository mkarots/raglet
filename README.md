<div align="center">
  <img src="assets/logo.png" alt="raglet logo" width="600">
</div>

# raglet

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Portable memory for small text corpora. No servers, no API keys, no infrastructure.**

There's a class of knowledge that's too big for a prompt but too small to justify a vector database: a codebase, a Slack export, a folder of meeting notes. raglet turns that text into a single directory you can save, git commit, or carry to another machine.

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

# Package it for sharing
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

---

## CLI

```bash
# Build a knowledge base
raglet build docs/ --out raglet-docs/
raglet build docs/ --out raglet-docs/ --chunk-size 1024 --model all-mpnet-base-v2

# Search it
raglet query "how does authentication work?" --raglet .raglet/
raglet query "what is X?" --raglet knowledge.sqlite --top-k 10

# Add files incrementally
raglet add new_file.txt --raglet .raglet/

# Convert between formats
raglet package --raglet .raglet/ --format zip --out export.zip
raglet package --raglet export.zip --format sqlite --out knowledge.sqlite
```

---

## Storage formats

raglet supports three formats. All can be loaded with `RAGlet.load()` — it auto-detects from the path.

| Format | Best for | Incremental updates |
|--------|----------|-------------------|
| `.raglet/` directory | Development, git-tracked knowledge bases | ✅ |
| `.sqlite` / `.db` | Single-file portability, production use | ✅ |
| `.zip` | Sharing, export/import | ❌ read-only |

```python
rag.save(".raglet/")          # directory
rag.save("knowledge.sqlite")  # SQLite
rag.save("export.zip")        # zip archive

rag = RAGlet.load(".raglet/")
rag = RAGlet.load("knowledge.sqlite")
rag = RAGlet.load("export.zip")
```

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

rag = RAGlet.load(".raglet/") if Path(".raglet/").exists() else RAGlet.from_files(["docs/"])
unsaved = 0

while True:
    query = input("You: ")
    if query == "exit":
        if unsaved: rag.save(".raglet/")
        break

    results = rag.search(query, top_k=5)
    response = your_llm(results, query)

    rag.add_text(query, source="chat")
    rag.add_text(response, source="chat")
    unsaved += len(query) + len(response)

    if unsaved >= 1000:
        rag.save(".raglet/")
        unsaved = 0
```

### Incremental updates

```python
# Add raw text
rag.add_text("Some text", source="manual")
rag.add_text("More context", source="chat", metadata={"session": "abc"})

# Add files
rag.add_file("new_doc.txt")
rag.add_files(["file1.txt", "file2.md"])

# Save with incremental flag
rag.save(".raglet/", incremental=True)
```

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

## Features

- ✅ Text extraction from `.txt` and `.md` files
- ✅ Sentence-aware chunking
- ✅ Local embeddings via sentence-transformers (no API keys)
- ✅ Vector search via FAISS
- ✅ Three portable formats: directory, SQLite, zip
- ✅ Incremental updates
- ✅ CLI interface
- ✅ Docker image

---

## Principles

**Portable** — One directory (or file). Git commit it, email it, load it on another machine.

**Small by design** — Workspace-scale: codebases, conversations, notes. Not the internet.

**Retrieval only** — raglet finds chunks. You decide what to do with them. Bring your own LLM.

**Open format** — JSON files you can read, edit, and extract. No proprietary format, no lock-in.

**Zero infrastructure** — `pip install raglet` or `docker run`. That's it.

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
- [Roadmap](docs/plans/FINAL_PLAN.md)

---

## License

MIT
