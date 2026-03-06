# Decision 006: Technology Stack

**Date:** December 2024  
**Status:** Accepted

## Context

Need local, CPU-friendly, zero-infrastructure stack.

## Decision

- **Embeddings:** sentence-transformers (local, CPU-friendly)
- **Vector Search:** FAISS IndexFlatL2 (no external DB)
- **Document Processing:** Text files only (UTF-8, no binary formats)
- **Config:** YAML/JSON files
- **No external services:** Everything works offline

## Rationale

- All local (no cloud APIs)
- CPU-friendly (works without GPU)
- Mature libraries
- Small footprint
- Zero infrastructure
- Text-only keeps dependencies minimal (no PDF/DOCX parsers)

## Consequences

- Works offline
- No API keys needed
- Slower than GPU (acceptable for workspace-scale)
- Larger dependencies (sentence-transformers ~80MB)

## Alternatives Considered

- OpenAI embeddings API (rejected - requires API key, infrastructure)
- Pinecone/Weaviate (rejected - requires cloud infrastructure)
- ChromaDB (rejected - requires local server)
