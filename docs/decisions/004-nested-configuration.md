# Decision 004: Nested Configuration Structure

**Date:** December 2024  
**Status:** Accepted

## Context

Need ergonomic configuration that can go deep without API complexity.

## Decision

Use nested configuration classes: `ChunkingConfig`, `EmbeddingConfig`, `SearchConfig`, `FileProcessingConfig` within `RAGletConfig`.

## Rationale

- Better organization (related settings grouped)
- Stronger validation (per-component)
- More flexible (override specific components)
- Reusable (compose config components)
- Self-documenting structure

## Consequences

- More structured configuration
- Can configure everything ergonomically
- Backward compatible (flat config auto-converts)
- YAML/JSON config files are nested

## Structure

```python
RAGletConfig
├── chunking: ChunkingConfig
├── embeddings: EmbeddingConfig
├── search: SearchConfig
└── file_processing: FileProcessingConfig
```
