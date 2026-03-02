<div align="center">
  <img src="assets/logo.png" alt="tinyrag logo" width="300">
</div>

# tinyrag

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Portable memory for small text corpora.**

tinyrag creates searchable `.tinyrag` files from your documents. No infrastructure, no servers, no API keys. Just `pip install tinyrag`.

## The Problem

There's a class of knowledge that's **small but too big for a prompt**:
- A codebase
- A Slack conversation
- A WhatsApp chat export
- A folder of meeting notes

These are small (a few megabytes) but don't fit in a context window. They also don't justify a vector database, server, or infrastructure setup.

## The Solution

tinyrag is **portable memory**. It takes small context and turns it into a single `.tinyrag` file that you can save, share, commit, or carry around. Load it anywhere, search it instantly, and get retrieval-ready context for any LLM or tool.

**No server. No API keys. No infrastructure. Just a Python object and a file.**

## Quick Start

```python
from tinyrag import TinyRAG

# Create from files
rag = TinyRAG.from_files(["doc.txt", "notes.md"])

# Get chunks
chunks = rag.get_all_chunks()

# Search (coming in Milestone 2)
# results = rag.search("what is X?", top_k=5)

# Save portable file (coming in Milestone 3)
# rag.save("knowledge.tinyrag")
```

## Installation

```bash
pip install tinyrag
```

For development (requires [uv](https://github.com/astral-sh/uv)):

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make install-dev
```

## Features

**Current (Milestone 1):**
- ✅ Extract text from .txt and .md files
- ✅ Intelligent chunking with sentence awareness
- ✅ SOLID architecture with clear interfaces

**Coming Soon:**
- 🔜 Portable `.tinyrag` file format
- 🔜 Local embeddings (sentence-transformers)
- 🔜 Vector search (FAISS)
- 🔜 Save/load operations
- 🔜 PDF, HTML, DOCX support

## Principles

1. **Portable** - One `.tinyrag` file. Save it, git commit it, email it
2. **Small by design** - Workspace-scale (codebases, conversations, notes). Not the internet
3. **Retrieval only** - tinyrag finds chunks. You decide what to do with them. Bring your own LLM
4. **Open format** - The `.tinyrag` file is easily decodable. Embeddings are extractable. No lock-in
5. **Zero infrastructure** - `pip install tinyrag`. That's it

## Development

```bash
make install-dev     # Install with dev dependencies
make test            # Run all tests
make test-unit       # Unit tests only
make test-integration # Integration tests only
make test-e2e        # E2E tests only
make lint            # Run linters
make format          # Format code
make type-check      # Type checking
make ci              # Full CI pipeline
```

## Architecture

tinyrag follows SOLID principles with clear separation of concerns:

- **core/** - Domain models and orchestrator
- **processing/** - Document extraction and chunking
- **embeddings/** - Embedding generation
- **vector_store/** - Vector storage and search
- **storage/** - File serialization
- **config/** - Configuration system

See [docs/proposals/ARCHITECTURE.md](docs/proposals/ARCHITECTURE.md) for details.

## Status

**Milestone 1 Complete** - Foundation & Core Structure  
**Milestone 2 In Progress** - Embeddings & Vector Store

See [plans/FINAL_PLAN.md](plans/FINAL_PLAN.md) for roadmap.

## Documentation

- [Problem Statement](docs/problems/00-problem-statement.md) - Why tinyrag exists
- [Architecture Decisions](docs/decisions/) - All architectural decisions
- [Implementation Plan](plans/FINAL_PLAN.md) - Roadmap and milestones
- [Agent Instructions](CLAUDE.md) - For contributors

## License

MIT
