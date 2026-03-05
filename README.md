<div align="center">
  <img src="assets/logo.png" alt="raglet logo" width="600">
</div>

# raglet

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Portable memory for small text corpora.**

raglet creates searchable `.raglet` files from your documents. No infrastructure, no servers, no API keys. Just `pip install raglet`.

## The Problem

There's a class of knowledge that's **small but too big for a prompt**:
- A codebase
- A Slack conversation
- A WhatsApp chat export
- A folder of meeting notes

These are small (a few megabytes) but don't fit in a context window. They also don't justify a vector database, server, or infrastructure setup.

## The Solution

raglet is **portable memory**. It takes small context and turns it into a single `.raglet` file that you can save, share, commit, or carry around. Load it anywhere, search it instantly, and get retrieval-ready context for any LLM or tool.

**No server. No API keys. No infrastructure. Just a Python object and a file.**

## Quick Start

### Python Library

```python
from raglet import RAGlet

# Create from files
rag = RAGlet.from_files(["doc.txt", "notes.md"])

# Search for relevant chunks
results = rag.search("what is X?", top_k=5)

# Get all chunks
chunks = rag.get_all_chunks()

# Save to directory (default format)
rag.save(".raglet/")

# Load later
rag = RAGlet.load(".raglet/")
```

### Docker CLI

**Use as a standalone tool:** Run raglet instantly against any workspace:

```bash
# Build knowledge base from workspace
# -v mounts your local project directory into the container at /workspace
# --workspace tells raglet where to find files inside the container
docker run -v /path/to/project:/workspace mkarots/raglet \
  --workspace /workspace build

# Query knowledge base
docker run -v /path/to/project:/workspace mkarots/raglet \
  --workspace /workspace query "what is Python?" --top-k 10

# Chat with Claude API (uses raglet context)
docker run -v /path/to/project:/workspace -e ANTHROPIC_API_KEY=your-key mkarots/raglet \
  --workspace /workspace chat "explain Python" --top-k 5

# Add files incrementally
docker run -v /path/to/project:/workspace mkarots/raglet \
  --workspace /workspace add new_file.txt

# Inspect knowledge base
docker run -v /path/to/project:/workspace mkarots/raglet \
  --workspace /workspace inspect

# Export to zip for sharing
docker run -v /path/to/project:/workspace mkarots/raglet \
  --workspace /workspace export --output knowledge.zip

# Use zip file directly (no need to extract)
# First, export to zip (if you haven't already):
docker run -v /path/to/project:/workspace mkarots/raglet \
  --workspace /workspace export --output knowledge.zip

# Then query using the zip file:
docker run -v /path/to/project:/workspace mkarots/raglet \
  --workspace /workspace query "your query" --kb-name knowledge.zip
```

**Knowledge base lives in `.raglet/` directory** - mount your workspace and it just works!

**Note:** If you encounter errors with zip files, make sure you're using the latest Docker image. Rebuild with `make docker-build` or pull the latest from Docker Hub.

## Installation

### Python Package

```bash
pip install raglet
```

### Docker Image

```bash
# Pull from Docker Hub
docker pull mkarots/raglet

# Or build locally
docker build -t mkarots/raglet .
```

### Development

For development (requires [uv](https://github.com/astral-sh/uv)):

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make install-dev
```

## Features

**Current:**
- ✅ Extract text from .txt and .md files
- ✅ Intelligent chunking with sentence awareness
- ✅ Local embeddings (sentence-transformers)
- ✅ Vector search (FAISS)
- ✅ Semantic search API
- ✅ Portable directory format (`.raglet/`)
- ✅ Save/load operations
- ✅ Incremental updates
- ✅ CLI interface
- ✅ Docker image
- ✅ SOLID architecture with clear interfaces

**Coming Soon:**
- 🔜 PDF, HTML, DOCX support
- 🔜 Zip export format

## Principles

1. **Portable** - One `.raglet/` directory. Save it, git commit it, email it (or export to zip)
2. **Small by design** - Workspace-scale (codebases, conversations, notes). Not the internet
3. **Retrieval only** - raglet finds chunks. You decide what to do with them. Bring your own LLM
4. **Open format** - The `.raglet/` directory is easily inspectable (JSON files). Embeddings are extractable. No lock-in
5. **Zero infrastructure** - `pip install raglet` or `docker run mkarots/raglet`. That's it

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

raglet follows SOLID principles with clear separation of concerns:

- **core/** - Domain models and orchestrator
- **processing/** - Document extraction and chunking
- **embeddings/** - Embedding generation
- **vector_store/** - Vector storage and search
- **storage/** - File serialization
- **config/** - Configuration system

See [docs/proposals/ARCHITECTURE.md](docs/proposals/ARCHITECTURE.md) for details.

## Status

**Milestone 3 Complete** - Portable File Format & CLI  
**Ready for Use** - Directory format, Docker image, CLI interface

See [plans/FINAL_PLAN.md](plans/FINAL_PLAN.md) for roadmap.

## Documentation

- [Problem Statement](docs/problems/00-problem-statement.md) - Why raglet exists
- [Architecture Decisions](docs/decisions/) - All architectural decisions
- [Implementation Plan](plans/FINAL_PLAN.md) - Roadmap and milestones
- [Agent Instructions](CLAUDE.md) - For contributors

## License

MIT
