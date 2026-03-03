# Decision 005: Portable .raglet File Format

**Date:** December 2024  
**Status:** Accepted

## Context

Need portable knowledge bases that can be saved, shared, and loaded anywhere.

## Decision

Single `.raglet` file as zip archive containing chunks, embeddings, index, and metadata.

## Rationale

- Portable (email, git commit, share)
- Reproducible (same file, same results)
- Version controllable
- No dependencies (file contains everything)
- Open format (decodable without library)

## Consequences

- Static files (can't update without recreating)
- Larger than raw text (embeddings + index)
- No real-time updates
- Core to value proposition

## Format

```
my_knowledge.raglet (zip)
├── metadata.json    # Config, version, creation date
├── chunks.json      # Chunk objects
├── embeddings.npy   # NumPy array
├── faiss_index.bin  # FAISS index
└── sources.json     # Source file metadata
```
