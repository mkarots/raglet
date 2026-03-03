# Milestone 2: Embeddings & Vector Store

**Status:** Planning  
**Goal:** Embedding generation and vector search working  
**Estimated Duration:** Week 2

---

## Overview

Milestone 2 implements the core retrieval capabilities: generating embeddings from chunks and storing/searching them using FAISS. This milestone completes the pipeline from files → chunks → embeddings → searchable index.

**Key Deliverables:**
- ✅ Generate embeddings from chunks using sentence-transformers
- ✅ Store vectors in FAISS index
- ✅ Store vectors in FAISS index
- ✅ Search vectors and retrieve relevant chunks
- ✅ Full pipeline working: files → chunks → embeddings → search

---

## Architecture Context

Following SOLID principles established in Milestone 1:

- **Dependency Inversion:** `TinyRAG` depends on `EmbeddingGenerator` and `VectorStore` interfaces, not concrete implementations
- **Single Responsibility:** Embedding generation is separate from vector storage/search
- **Open/Closed:** New embedding models or vector stores can be added without modifying existing code

---

## Tasks Breakdown

### 0. Interface Definitions (Prerequisites)

**Files:** 
- `tinyrag/embeddings/interfaces.py`
- `tinyrag/vector_store/interfaces.py`

**Requirements:**
- [ ] Create `embeddings/interfaces.py` with `EmbeddingGenerator` interface
  - [ ] `generate(chunks: List[Chunk]) -> np.ndarray` - Generate embeddings for chunks
  - [ ] `generate_single(text: str) -> np.ndarray` - Generate embedding for single text
  - [ ] `get_dimension() -> int` - Return embedding dimension
- [ ] Create `vector_store/interfaces.py` with `VectorStore` interface
  - [ ] `add_vectors(vectors: np.ndarray, chunks: List[Chunk]) -> None` - Add vectors with chunks
  - [ ] `search(query_vector: np.ndarray, top_k: int) -> List[Chunk]` - Search and return chunks with scores
  - [ ] `get_count() -> int` - Return number of vectors stored

**Testing:**
- [ ] Verify interfaces can be imported
- [ ] Verify abstract methods are properly defined

---

### 1. Embedding Generator Implementation

**File:** `tinyrag/embeddings/generator.py`

**Interface:** `tinyrag/embeddings/interfaces.py` (created in step 0)

**Implementation: `SentenceTransformerGenerator`**

**Requirements:**
- [ ] Implement `EmbeddingGenerator` interface
- [ ] Use `sentence-transformers` library
- [ ] Default model: `all-MiniLM-L6-v2` (384 dimensions, CPU-friendly)
- [ ] Batch processing: Process 32 chunks at a time
- [ ] Model loading and caching (load once, reuse)
- [ ] Implement `generate()` method: `List[Chunk] → np.ndarray`
- [ ] Implement `generate_single()` method: `str → np.ndarray` (for queries)
- [ ] Implement `get_dimension()` method: Returns embedding dimension
- [ ] Handle device selection (CPU default, GPU optional)
- [ ] Error handling for model loading failures

**Configuration:**
- Model name (configurable)
- Batch size (default: 32)
- Device (CPU/GPU)
- Normalize embeddings (optional, for cosine similarity)

**Dependencies:**
- `sentence-transformers>=2.2.0`
- `numpy>=1.24.0`
- `torch` (via sentence-transformers)

**Testing:**
- [ ] Unit test: Generate embeddings for single chunk
- [ ] Unit test: Generate embeddings for batch of chunks
- [ ] Unit test: Verify embedding dimensions match model
- [ ] Unit test: Batch processing works correctly
- [ ] Unit test: Model caching (loads once)
- [ ] Unit test: Error handling for invalid model name
- [ ] Integration test: Generate embeddings for real chunks

**Example Usage:**
```python
from tinyrag.embeddings.generator import SentenceTransformerGenerator
from tinyrag.config.config import EmbeddingConfig

config = EmbeddingConfig(
    model="all-MiniLM-L6-v2",
    batch_size=32,
    device="cpu"
)
generator = SentenceTransformerGenerator(config)

chunks = [...]  # List[Chunk]
embeddings = generator.generate(chunks)  # np.ndarray shape (len(chunks), 384)
```

---

### 2. Vector Store Implementation

**File:** `tinyrag/vector_store/faiss_store.py`

**Interface:** `tinyrag/vector_store/interfaces.py` (already defined in Milestone 1)

**Implementation: `FAISSVectorStore`**

**Requirements:**
- [ ] Implement `VectorStore` interface
- [ ] Use FAISS `IndexFlatL2` (exact L2 distance, simple and fast)
- [ ] Initialize index with embedding dimension
- [ ] Implement `add_vectors()` method: Add embeddings with metadata
- [ ] Implement `search()` method: Query vector → top_k results with scores
- [ ] Store chunk metadata alongside vectors (for retrieval)
- [ ] Handle empty index (no vectors added yet)
- [ ] Handle search when index is empty
- [ ] Index persistence (save/load FAISS index to binary)

**Vector-Chunk Mapping:**
- Need to map vector indices to chunk objects
- Store chunks separately, use index for similarity search
- Return chunks with similarity scores

**Configuration:**
- Index type (default: IndexFlatL2)
- Distance metric (L2 default, can extend to cosine)
- Max vectors (optional limit)

**Dependencies:**
- `faiss-cpu>=1.7.4` (or `faiss-gpu` for GPU support)
- `numpy>=1.24.0`

**Testing:**
- [ ] Unit test: Initialize empty index
- [ ] Unit test: Add vectors to index
- [ ] Unit test: Search returns top_k results
- [ ] Unit test: Search scores are correct (L2 distance)
- [ ] Unit test: Search with empty index (returns empty)
- [ ] Unit test: Retrieve chunks with search results
- [ ] Unit test: Index persistence (save/load)
- [ ] Integration test: Add embeddings → search → verify results

**Example Usage:**
```python
from tinyrag.vector_store.faiss_store import FAISSVectorStore
from tinyrag.config.config import SearchConfig

config = SearchConfig(
    default_top_k=5,
    similarity_threshold=None
)
store = FAISSVectorStore(embedding_dim=384, config=config)

# Add vectors with chunks
embeddings = np.array([...])  # shape (n_chunks, 384)
chunks = [...]  # List[Chunk]
store.add_vectors(embeddings, chunks)

# Search
query_embedding = np.array([...])  # shape (384,)
results = store.search(query_embedding, top_k=5)
# Returns: List[Chunk] with score attribute set
```

---

### 3. Configuration Extension

**File:** `tinyrag/config/config.py`

**Extend existing config classes:**

**New: `EmbeddingConfig`**
- [ ] `model: str = "all-MiniLM-L6-v2"` - Embedding model name
- [ ] `batch_size: int = 32` - Batch size for processing
- [ ] `device: str = "cpu"` - Device (cpu/cuda)
- [ ] `normalize: bool = False` - Normalize embeddings (for cosine similarity)
- [ ] `validate()` method - Validate model name, batch_size > 0, etc.

**New: `SearchConfig`**
- [ ] `default_top_k: int = 5` - Default number of results
- [ ] `similarity_threshold: Optional[float] = None` - Minimum similarity score
- [ ] `index_type: str = "flat_l2"` - FAISS index type
- [ ] `validate()` method - Validate top_k > 0, threshold in [0, 1] if set

**Update: `TinyRAGConfig`**
- [ ] Add `embedding: EmbeddingConfig` field
- [ ] Add `search: SearchConfig` field
- [ ] Update `validate()` to validate nested configs
- [ ] Update `to_dict()` and `from_dict()` for nested serialization

**Testing:**
- [ ] Unit test: EmbeddingConfig validation
- [ ] Unit test: SearchConfig validation
- [ ] Unit test: TinyRAGConfig with nested configs
- [ ] Unit test: Config serialization/deserialization

**Example:**
```python
from tinyrag.config.config import TinyRAGConfig, EmbeddingConfig, SearchConfig

config = TinyRAGConfig(
    chunking=ChunkingConfig(size=512, overlap=50),
    embedding=EmbeddingConfig(
        model="all-MiniLM-L6-v2",
        batch_size=32,
        device="cpu"
    ),
    search=SearchConfig(
        default_top_k=5,
        similarity_threshold=None
    )
)
```

---

### 4. Core Integration

**File:** `tinyrag/core/rag.py`

**Complete `TinyRAG.from_files()` implementation:**

**Requirements:**
- [ ] Wire up full pipeline: Extract → Chunk → Embed → Index
- [ ] Use dependency injection (interfaces, not concrete classes)
- [ ] Create default implementations via factories if not provided
- [ ] Handle errors gracefully at each step
- [ ] Store chunks, embeddings, and index in TinyRAG instance
- [ ] Return fully initialized TinyRAG with searchable index

**Pipeline Flow:**
1. Extract text from files (using `DocumentExtractor`)
2. Chunk text (using `Chunker`)
3. Generate embeddings (using `EmbeddingGenerator`)
4. Add to vector store (using `VectorStore`)
5. Store references in TinyRAG instance

**Dependencies:**
- `TinyRAG` receives `EmbeddingGenerator` and `VectorStore` via constructor or factory
- Factory functions create default implementations if not provided
- All components use interfaces (Dependency Inversion Principle)

**Error Handling:**
- File extraction errors (skip file, log warning)
- Chunking errors (fallback to simple chunking)
- Embedding errors (fail fast with clear message)
- Vector store errors (fail fast with clear message)

**Testing:**
- [ ] Integration test: Full pipeline with real files
- [ ] Integration test: Custom embedding generator (mock)
- [ ] Integration test: Custom vector store (mock)
- [ ] Integration test: Error handling at each step
- [ ] E2E test: Create TinyRAG → Search → Verify results

**Example:**
```python
from tinyrag import TinyRAG
from tinyrag.config.config import TinyRAGConfig

# Simple usage (uses defaults)
rag = TinyRAG.from_files(["doc1.txt", "doc2.md"])

# With custom config
config = TinyRAGConfig(
    embedding=EmbeddingConfig(model="all-mpnet-base-v2"),
    search=SearchConfig(default_top_k=10)
)
rag = TinyRAG.from_files(["doc1.txt"], config=config)

# With custom components (dependency injection)
from tinyrag.embeddings.generator import SentenceTransformerGenerator
from tinyrag.vector_store.faiss_store import FAISSVectorStore

generator = SentenceTransformerGenerator(config.embedding)
store = FAISSVectorStore(embedding_dim=generator.get_dimension(), config=config.search)

rag = TinyRAG.from_files(
    ["doc1.txt"],
    embedding_generator=generator,
    vector_store=store,
    config=config
)
```

---

### 5. Search API Implementation

**File:** `tinyrag/core/rag.py`

**Add `search()` method to `TinyRAG`:**

**Requirements:**
- [ ] Accept query string (not just vector)
- [ ] Embed query using `EmbeddingGenerator`
- [ ] Search using `VectorStore`
- [ ] Return `List[Chunk]` with scores
- [ ] Use `default_top_k` from config if not specified
- [ ] Apply `similarity_threshold` if configured
- [ ] Handle empty index (return empty list)

**Method Signature:**
```python
def search(
    self,
    query: str,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None
) -> List[Chunk]:
    """Search and retrieve relevant chunks.
    
    Args:
        query: Search query string
        top_k: Number of results (uses config default if None)
        similarity_threshold: Minimum similarity score (uses config default if None)
    
    Returns:
        List of Chunk objects with score attribute set
    """
```

**Testing:**
- [ ] Unit test: Search with query string
- [ ] Unit test: Search respects top_k parameter
- [ ] Unit test: Search respects similarity_threshold
- [ ] Unit test: Search uses config defaults
- [ ] Unit test: Search with empty index
- [ ] Integration test: Search returns relevant results
- [ ] E2E test: Create → Search → Verify relevance

**Example:**
```python
rag = TinyRAG.from_files(["doc1.txt", "doc2.md"])

# Simple search
results = rag.search("what is X?")

# With parameters
results = rag.search("what is X?", top_k=10, similarity_threshold=0.7)

# Results have score attribute
for chunk in results:
    print(f"Score: {chunk.score}")
    print(f"Text: {chunk.text}")
    print(f"Source: {chunk.source}")
```

---

## Implementation Order

**Recommended sequence:**

1. **EmbeddingConfig** - Define configuration structure first
2. **SentenceTransformerGenerator** - Implement embedding generation
3. **SearchConfig** - Define search configuration
4. **FAISSVectorStore** - Implement vector storage/search
5. **TinyRAGConfig extension** - Wire up nested configs
6. **TinyRAG.from_files()** - Complete pipeline integration
7. **TinyRAG.search()** - Add search API

**Rationale:**
- Config first (defines contracts)
- Embeddings before vector store (vector store needs dimension)
- Vector store before integration (integration needs both)
- Search API last (depends on everything)

---

## Testing Strategy

### Unit Tests

**Location:** `tests/unit/`

- [ ] `test_embeddings.py` - Test `SentenceTransformerGenerator`
- [ ] `test_vector_store.py` - Test `FAISSVectorStore`
- [ ] `test_config.py` - Test `EmbeddingConfig` and `SearchConfig`

**Coverage Goals:**
- >90% coverage for embedding generator
- >90% coverage for vector store
- >80% coverage for config classes

### Integration Tests

**Location:** `tests/integration/`

- [ ] `test_embedding_pipeline.py` - Chunks → Embeddings flow
- [ ] `test_vector_search.py` - Embeddings → Search flow
- [ ] `test_full_pipeline.py` - Files → Chunks → Embeddings → Search

**Test Scenarios:**
- Small dataset (5 files, ~50 chunks)
- Medium dataset (20 files, ~200 chunks)
- Empty dataset (edge case)
- Single chunk (edge case)

### E2E Tests

**Location:** `tests/e2e/`

- [ ] `test_e2e_search.py` - Full workflow: create → search → verify

**Test Cases:**
- Create TinyRAG from files
- Search for specific content
- Verify top result is relevant
- Verify scores are reasonable

---

## Dependencies to Add

**Update `pyproject.toml`:**

```toml
dependencies = [
    "numpy>=1.24.0",
    "sentence-transformers>=2.2.0",  # NEW
    "faiss-cpu>=1.7.4",              # NEW
]
```

**Optional:**
```toml
[project.optional-dependencies]
gpu = [
    "faiss-gpu>=1.7.4",  # GPU-accelerated FAISS
]
```

---

## Success Criteria

**Functional:**
- ✅ Can generate embeddings for chunks
- ✅ Can store embeddings in FAISS index
- ✅ Can search and retrieve relevant chunks
- ✅ Full pipeline works: files → chunks → embeddings → search
- ✅ Search returns relevant results (top-1 is correct >80% of time)

**Performance:**
- ✅ Embedding generation: <1 second for 100 chunks
- ✅ Search: <100ms for top-5 results
- ✅ Memory: <500MB for 1000 chunks

**Quality:**
- ✅ Unit test coverage >80%
- ✅ Integration tests pass
- ✅ E2E tests pass
- ✅ Error handling is comprehensive
- ✅ Code follows SOLID principles

---

## Risks & Mitigations

**Risk 1: Model download on first run**
- **Mitigation:** Document model download, consider caching strategy
- **Impact:** First run slower, but acceptable

**Risk 2: FAISS index size**
- **Mitigation:** Monitor file sizes, optimize if needed
- **Impact:** Should be fine for workspace-scale (<10MB for 1000 chunks)

**Risk 3: Embedding quality**
- **Mitigation:** Use proven model (all-MiniLM-L6-v2), allow model override
- **Impact:** May need better model for some use cases (configurable)

**Risk 4: Memory usage**
- **Mitigation:** Batch processing, monitor memory in tests
- **Impact:** Should be fine for workspace-scale

---

## Documentation Updates

**Update README.md:**
- [ ] Add search example to Quick Start
- [ ] Document embedding configuration
- [ ] Document search configuration

**Create/Update:**
- [ ] `docs/API.md` - Document `TinyRAG.search()` method
- [ ] `docs/ARCHITECTURE.md` - Document embedding and vector store components

---

## Deliverables Checklist

- [ ] `SentenceTransformerGenerator` implemented and tested
- [ ] `FAISSVectorStore` implemented and tested
- [ ] `EmbeddingConfig` and `SearchConfig` implemented
- [ ] `TinyRAGConfig` extended with nested configs
- [ ] `TinyRAG.from_files()` completes full pipeline
- [ ] `TinyRAG.search()` method implemented
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] E2E tests written and passing
- [ ] Documentation updated
- [ ] Dependencies added to `pyproject.toml`

---

## Next Steps After Milestone 2

Once Milestone 2 is complete, we'll have:
- ✅ Full retrieval pipeline working
- ✅ Searchable TinyRAG instances
- ✅ Foundation for Milestone 3 (file format)

**Milestone 3 will add:**
- Save/load `.tinyrag` files
- Serialize embeddings and FAISS index
- Portable file format

---

**Status:** Ready for Implementation  
**Last Updated:** December 2024
