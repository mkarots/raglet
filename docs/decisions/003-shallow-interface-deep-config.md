# Decision 003: Shallow Interface, Deep Configuration

**Date:** December 2024  
**Status:** Accepted

## Context

Need to balance simplicity for 80% of users with power for 20% of users.

## Decision

**Shallow Interface:** Simple API that "just works"  
**Deep Configuration:** Ergonomic, sufficient configuration system

## Rationale

- API stays simple: `RAGlet.from_files(["doc.txt"])`
- No nested parameters in constructor
- Configuration is escape hatch for deep customization
- Progressive disclosure: simple → override → preset → deep config

## Consequences

- Simple API for common use cases
- Deep configuration for advanced use cases
- No API bloat (complex options in config, not API)
- Configuration is portable (files, not code)

## Example

```python
# Shallow interface
rag = RAGlet.from_files(["doc.txt"])

# Deep configuration
config = RAGletConfig(
    chunking=ChunkingConfig(size=1024, strategy="semantic"),
    embeddings=EmbeddingConfig(model="all-mpnet-base-v2")
)
rag = RAGlet.from_files(["doc.txt"], config=config)
```
