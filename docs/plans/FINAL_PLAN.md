# tinyrag Final Implementation Plan

**Status:** Approved for Implementation  
**Date:** December 2024  
**Version:** 1.0 - Final

---

## Purpose of This Document

This is the **definitive plan** for tinyrag. It consolidates all decisions, sets clear boundaries, and defines exactly what we're building. **No scope changes without updating this document.**

---

## 1. Project Vision (from WHY.md)

### The Problem

There's a class of knowledge that's **small but too big for a prompt**:
- A codebase
- A Slack conversation
- A WhatsApp chat export
- A folder of meeting notes

These are small (a few megabytes) but don't fit in a context window. They also don't justify a vector database, server, or infrastructure setup.

### The Solution

**tinyrag is portable memory.**

A Python library that creates a single `.tinyrag` file containing:
- Text chunks
- Embeddings
- Vector index
- Metadata

**No server. No API keys. No infrastructure. Just `pip install tinyrag`.**

### Core Principles (Non-Negotiable)

1. **Portable** - One `.tinyrag` file. Save it, git commit it, email it, drag it to another machine
2. **Small by design** - Built for workspace-scale problems (codebases, conversations, notes). **Not the internet**
3. **Retrieval only** - tinyrag finds chunks. You decide what to do with them. **Bring your own LLM**
4. **Open format** - The `.tinyrag` file is easily decodable. Embeddings are extractable. No lock-in
5. **Zero infrastructure** - `pip install tinyrag`. That's it

---

## 2. What We're Building

### Core Product: Python Library

**Primary API:**
```python
from tinyrag import TinyRAG

# Create from files
rag = TinyRAG.from_files(["doc.txt", "notes.md"])

# Save portable file
rag.save("knowledge.tinyrag")

# Load later
rag = TinyRAG.load("knowledge.tinyrag")

# Search
results = rag.search("what is X?", top_k=5)
```

**What it does:**
- ✅ Processes text files (.txt, .md, .pdf, .html, .docx)
- ✅ Chunks text intelligently
- ✅ Generates embeddings locally
- ✅ Creates vector index (FAISS)
- ✅ Saves to portable `.tinyrag` file
- ✅ Loads `.tinyrag` files
- ✅ Searches and retrieves relevant chunks

**What it does NOT do:**
- ❌ No LLM integration (retrieval only)
- ❌ No web service/API
- ❌ No real-time updates (files are static)
- ❌ No large-scale datasets (workspace-scale only)
- ❌ No streaming/chunked loading (MVP)

---

## 3. Scope Boundaries

### ✅ IN SCOPE (MVP)

**Core Functionality:**
- Document processing: .txt, .md, .pdf, .html, .docx
- Text chunking (512 tokens, 50 overlap, sentence-aware)
- Local embeddings (sentence-transformers, CPU-friendly)
- Vector search (FAISS IndexFlatL2)
- Portable file format (.tinyrag zip archive)
- Save/load operations
- Search/retrieval API
- Configuration system (constructor params, config object, presets)
- Agent tools (Anthropic integration)

**File Format:**
- Zip archive containing: metadata.json, chunks.json, embeddings.npy, faiss_index.bin, sources.json
- Open format (decodable without library)
- Stores config used to create file

**Configuration:**
- Constructor parameters with defaults
- TinyRAGConfig object (reusable)
- Presets ("codebase", "documents", "conversations")
- Config file support (YAML) - optional
- Precedence: Constructor > Config object > Config file > Env vars > Defaults

**Agent Integration:**
- Three Anthropic tools: create_tinyrag, search_tinyrag, get_tinyrag_info
- Tool functions that wrap TinyRAG API
- Zero infrastructure (library, not API)

### ❌ OUT OF SCOPE (MVP)

**Explicitly NOT building:**
- ❌ Web service/API server
- ❌ React frontend or UI
- ❌ LLM integration (OpenAI, Anthropic, etc.)
- ❌ Real-time document updates
- ❌ Incremental updates to .tinyrag files
- ❌ Streaming/chunked loading
- ❌ Multiple embedding models per file
- ❌ Custom chunker plugins
- ❌ CLI tools (future enhancement)
- ❌ Large-scale datasets (>10MB per file, >1000 files)
- ❌ Production RAG pipeline features
- ❌ Database backends
- ❌ Authentication/authorization
- ❌ Rate limiting
- ❌ Monitoring/logging infrastructure

**Future Considerations (Post-MVP):**
- CLI tools
- Streaming support
- Custom chunkers
- Additional embedding models
- Incremental updates (if needed)

---

## 4. Technical Specifications

### Technology Stack

**Core Dependencies:**
- `sentence-transformers>=2.2.0` - Embeddings (local, CPU-friendly)
- `faiss-cpu>=1.7.4` - Vector search (no external DB)
- `numpy>=1.24.0` - Array operations

**Document Processing:**
- `PyPDF2>=3.0.0` - PDF extraction
- `beautifulsoup4>=4.12.0` - HTML parsing
- `python-docx>=1.1.0` - DOCX extraction
- `markdown>=3.5.0` - Markdown parsing

**Optional:**
- `faiss-gpu` - GPU acceleration (optional dependency)
- `pyyaml>=6.0` - Config file support

**No External Services:**
- ❌ No cloud APIs
- ❌ No databases
- ❌ No servers
- ❌ No authentication services

### Default Configuration

**Chunking:**
- Size: 512 tokens
- Overlap: 50 tokens
- Strategy: Sentence-aware

**Embeddings:**
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Batch size: 32
- Device: CPU (default)

**Search:**
- Default top_k: 5
- Similarity threshold: None (no filtering)
- Index type: FAISS IndexFlatL2 (exact L2 distance)

**File Limits:**
- Max file size: 10MB per file
- Max files: No hard limit (workspace-scale expected)
- Max chunks: No hard limit (but file size will be limiting factor)

### File Format Specification

**Structure:**
```
my_knowledge.tinyrag (zip archive)
├── metadata.json          # Version, config, creation date, chunk count
├── chunks.json            # Array of chunk objects with text, source, index, metadata
├── embeddings.npy         # NumPy float32 array (chunk_count, embedding_dim)
├── faiss_index.bin        # Serialized FAISS IndexFlatL2
└── sources.json           # Source file metadata (paths, sizes, types)
```

**Metadata Schema:**
```json
{
  "version": "1.0.0",
  "created_at": "2024-12-04T12:00:00Z",
  "config": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384
  },
  "chunk_count": 150,
  "source_count": 5
}
```

**Open Format Requirements:**
- Can extract embeddings independently
- Can decode chunks without library
- Can inspect metadata
- No proprietary encoding

---

## 5. API Specification

### Core API

```python
class TinyRAG:
    @classmethod
    def from_files(
        cls,
        files: List[str],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        default_top_k: int = 5,
        config: Optional["TinyRAGConfig"] = None,
        config_file: Optional[str] = None,
        preset: Optional[str] = None,
        **kwargs
    ) -> "TinyRAG":
        """Create TinyRAG from files."""
    
    def save(self, path: str) -> None:
        """Save to .tinyrag file."""
    
    @classmethod
    def load(cls, path: str) -> "TinyRAG":
        """Load from .tinyrag file."""
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List["Chunk"]:
        """Search and retrieve relevant chunks."""
    
    def get_all_chunks(self) -> List["Chunk"]:
        """Get all chunks (for context stuffing fallback)."""
```

### Configuration API

```python
@dataclass
class TinyRAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunk_strategy: str = "sentence-aware"
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    default_top_k: int = 5
    similarity_threshold: Optional[float] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def preset(cls, name: str) -> "TinyRAGConfig":
        """Get preset: 'codebase', 'documents', 'conversations'"""
    
    @classmethod
    def load(cls, path: str) -> "TinyRAGConfig":
        """Load from YAML/JSON file."""
    
    def save(self, path: str) -> None:
        """Save to file."""
    
    def merge(self, other: "TinyRAGConfig") -> "TinyRAGConfig":
        """Merge with another config."""
    
    def validate(self) -> None:
        """Validate config values."""
```

### Chunk Object

```python
@dataclass
class Chunk:
    text: str
    source: str              # Original file path
    index: int              # Position in document
    metadata: Dict[str, Any] # Custom metadata
    score: Optional[float] = None  # Similarity score (from search)
```

### Agent Tools API

```python
# tinyrag/tools.py

def create_tinyrag(
    file_paths: List[str],
    output_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> Dict[str, Any]:
    """Create .tinyrag file from documents."""

def search_tinyrag(
    file_path: str,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Search a .tinyrag file."""

def get_tinyrag_info(file_path: str) -> Dict[str, Any]:
    """Get metadata about a .tinyrag file."""
```

---

## 6. Architecture & Design Principles

### SOLID Architecture

Following SOLID principles for clarity and maintainability:

**Single Responsibility:**
- Each class/module has one clear purpose
- `TinyRAG` orchestrates (doesn't implement)
- `Chunker` chunks (doesn't extract or embed)
- `EmbeddingGenerator` generates embeddings (doesn't search)
- `VectorStore` stores/searches (doesn't generate embeddings)

**Open/Closed:**
- Open for extension (new extractors, generators, stores)
- Closed for modification (interfaces stable)

**Liskov Substitution:**
- Implementations are substitutable
- Any `VectorStore` implementation works
- Any `EmbeddingGenerator` implementation works

**Interface Segregation:**
- Small, focused interfaces
- Clients depend only on what they use

**Dependency Inversion:**
- Depend on abstractions (interfaces), not concretions
- `TinyRAG` depends on interfaces, not concrete classes

### Component Structure

```
tinyrag/
├── core/                    # Core domain logic
│   ├── rag.py              # TinyRAG orchestrator
│   └── chunk.py             # Chunk domain model
├── processing/              # Document processing
│   ├── extractors/          # File type extractors
│   ├── chunker.py          # Text chunking
│   └── interfaces.py       # Processing interfaces
├── embeddings/              # Embedding generation
│   ├── generator.py        # Embedding generator
│   └── interfaces.py       # Embedding interfaces
├── vector_store/            # Vector storage & search
│   ├── faiss_store.py      # FAISS implementation
│   └── interfaces.py       # Vector store interfaces
├── storage/                 # File format & persistence
│   ├── serializer.py       # .tinyrag serialization
│   ├── deserializer.py     # .tinyrag deserialization
│   └── interfaces.py       # Storage interfaces
├── config/                  # Configuration system
│   ├── config.py           # Configuration classes
│   ├── validators.py       # Configuration validation
│   └── loaders.py          # Config loading (YAML, JSON)
└── tools/                   # Agent tools
    └── agent_tools.py      # Anthropic tool functions
```

### Configuration Philosophy: "Shallow Interface, Deep Configuration"

**Shallow Interface (API):**
- Simple API that "just works" for 80% of users
- No nested parameters in constructor
- Common use cases work with defaults

**Deep Configuration (Escape Hatch):**
- When you need customization, configuration goes deep
- Ergonomic, sufficient, and powerful
- Every aspect is configurable

**Progressive Disclosure:**
1. Simple: `TinyRAG.from_files(["doc.txt"])`
2. Override: `chunk_size=1024`
3. Preset: `preset="codebase"`
4. Deep: `config=TinyRAGConfig(...)`

---

## 7. Implementation Phases

### Milestone 1: Foundation & Core Structure (Week 1)
**Goal:** SOLID architecture foundation with basic document processing

**Architecture Setup:**
- [ ] Set up Python package structure following SOLID architecture
  - [ ] Create directory structure (core/, processing/, embeddings/, etc.)
  - [ ] Set up `setup.py` and `pyproject.toml`
  - [ ] Define package entry points

**Interface Definitions:**
- [ ] Create `processing/interfaces.py` with `DocumentExtractor` interface
- [ ] Create `processing/interfaces.py` with `Chunker` interface
- [ ] Create `embeddings/interfaces.py` with `EmbeddingGenerator` interface
- [ ] Create `vector_store/interfaces.py` with `VectorStore` interface
- [ ] Create `storage/interfaces.py` with `RAGSerializer` and `RAGDeserializer` interfaces

**Core Domain Models:**
- [ ] Implement `core/chunk.py` - `Chunk` dataclass
  - [ ] text, source, index, metadata fields
  - [ ] to_dict() method
- [ ] Create `core/rag.py` skeleton - `TinyRAG` class structure
  - [ ] Define constructor with dependencies (interfaces)
  - [ ] Define from_files() signature (not implementation yet)

**Document Processing:**
- [ ] Implement `processing/extractors/text_extractor.py`
  - [ ] Implements `DocumentExtractor` interface
  - [ ] Handles .txt files
- [ ] Implement `processing/extractors/markdown_extractor.py`
  - [ ] Implements `DocumentExtractor` interface
  - [ ] Handles .md files
- [ ] Create `processing/extractor_factory.py`
  - [ ] Factory function to select appropriate extractor

**Chunking:**
- [ ] Implement `processing/chunker.py`
  - [ ] Implements `Chunker` interface
  - [ ] Basic chunking (512 tokens, 50 overlap)
  - [ ] Sentence-aware strategy
  - [ ] Preserves metadata

**Configuration (Basic):**
- [ ] Create `config/config.py` with basic structure
  - [ ] `ChunkingConfig` dataclass (size, overlap, strategy)
  - [ ] `TinyRAGConfig` dataclass (holds ChunkingConfig)
  - [ ] Basic validation methods

**Testing:**
- [ ] Unit tests for `Chunk` model
- [ ] Unit tests for `TextExtractor`
- [ ] Unit tests for `MarkdownExtractor`
- [ ] Unit tests for `Chunker`
- [ ] Integration test: Extract → Chunk flow

**Deliverable:** 
- ✅ SOLID architecture foundation established
- ✅ Can extract text from .txt and .md files
- ✅ Can chunk text into `Chunk` objects
- ✅ Interfaces defined for all components
- ✅ Basic configuration structure in place

### Milestone 2: Embeddings & Vector Store (Week 2)
**Goal:** Embedding generation and vector search working

**Embeddings:**
- [ ] Implement `embeddings/generator.py` - `SentenceTransformerGenerator`
  - [ ] Implements `EmbeddingGenerator` interface
  - [ ] Integrate sentence-transformers library
  - [ ] Batch processing (32 chunks at a time)
  - [ ] Model loading and caching
  - [ ] get_dimension() method

**Vector Store:**
- [ ] Implement `vector_store/faiss_store.py` - `FAISSVectorStore`
  - [ ] Implements `VectorStore` interface
  - [ ] FAISS IndexFlatL2 initialization
  - [ ] add_vectors() method
  - [ ] search() method with top_k
  - [ ] Handle metadata storage

**Configuration (Extended):**
- [ ] Extend `config/config.py`
  - [ ] `EmbeddingConfig` dataclass (model, batch_size, device)
  - [ ] `SearchConfig` dataclass (default_top_k, similarity_threshold)
  - [ ] Update `TinyRAGConfig` to include nested configs
  - [ ] Validation for embedding and search configs

**Core Integration:**
- [ ] Complete `TinyRAG.from_files()` implementation
  - [ ] Wire up extractors → chunker → embedding generator → vector store
  - [ ] Use interfaces (dependency injection)
  - [ ] Handle errors gracefully

**Testing:**
- [ ] Unit tests for `SentenceTransformerGenerator`
- [ ] Unit tests for `FAISSVectorStore`
- [ ] Integration test: Extract → Chunk → Embed → Index flow
- [ ] Mock tests for `TinyRAG.from_files()` with interfaces

**Deliverable:**
- ✅ Can generate embeddings from chunks
- ✅ Can store vectors in FAISS index
- ✅ Can search vectors and retrieve chunks
- ✅ Full pipeline working: files → chunks → embeddings → search

### Milestone 3: Portable File Format (Week 3)
**Goal:** Save and load .tinyrag files

**Serialization:**
- [ ] Implement `storage/serializer.py` - `TinyRAGSerializer`
  - [ ] Implements `RAGSerializer` interface
  - [ ] Serialize chunks to JSON
  - [ ] Serialize embeddings to NumPy .npy format
  - [ ] Serialize FAISS index to binary
  - [ ] Create zip archive with all components
  - [ ] Store config in metadata.json
  - [ ] Version handling

**Deserialization:**
- [ ] Implement `storage/deserializer.py` - `TinyRAGDeserializer`
  - [ ] Implements `RAGDeserializer` interface
  - [ ] Extract zip archive
  - [ ] Load chunks from JSON
  - [ ] Load embeddings from .npy
  - [ ] Load FAISS index from binary
  - [ ] Reconstruct TinyRAG instance
  - [ ] Version compatibility checks

**Configuration (Deep):**
- [ ] Implement deep configuration structure
  - [ ] Nested config classes (ChunkingConfig, EmbeddingConfig, SearchConfig, FileProcessingConfig)
  - [ ] `TinyRAGConfig.from_dict()` for nested loading
  - [ ] `TinyRAGConfig.to_dict()` for serialization
  - [ ] Config validation for all nested configs
  - [ ] Store full config in .tinyrag metadata

**Core Methods:**
- [ ] Implement `TinyRAG.save()` method
  - [ ] Use `RAGSerializer` interface
  - [ ] Factory pattern for serializer
- [ ] Implement `TinyRAG.load()` classmethod
  - [ ] Use `RAGDeserializer` interface
  - [ ] Factory pattern for deserializer
  - [ ] Reconstruct all dependencies

**Testing:**
- [ ] Unit tests for `TinyRAGSerializer`
- [ ] Unit tests for `TinyRAGDeserializer`
- [ ] Integration test: Save → Load → Verify
- [ ] Test version compatibility
- [ ] Test config serialization/deserialization

**Deliverable:**
- ✅ Can save TinyRAG to .tinyrag file
- ✅ Can load .tinyrag file to TinyRAG
- ✅ File format is open and decodable
- ✅ Config stored and restored correctly
- ✅ Version handling works

### Milestone 4: Extended Document Support (Week 4)
**Goal:** Support all target file formats

**Document Extractors (Following OCP):**
- [ ] Implement `processing/extractors/pdf_extractor.py`
  - [ ] Implements `DocumentExtractor` interface
  - [ ] Uses PyPDF2 for extraction
  - [ ] Error handling for corrupted PDFs
- [ ] Implement `processing/extractors/html_extractor.py`
  - [ ] Implements `DocumentExtractor` interface
  - [ ] Uses BeautifulSoup for parsing
  - [ ] Removes scripts/styles (configurable)
- [ ] Implement `processing/extractors/docx_extractor.py`
  - [ ] Implements `DocumentExtractor` interface
  - [ ] Uses python-docx for extraction
  - [ ] Handles formatting

**Extractor Factory:**
- [ ] Update `processing/extractor_factory.py`
  - [ ] Auto-detect file type
  - [ ] Return appropriate extractor
  - [ ] Handle unsupported file types

**Configuration (File Processing):**
- [ ] Complete `FileProcessingConfig` implementation
  - [ ] Per-file-type settings (text_extraction dict)
  - [ ] max_file_size_mb, supported_extensions
  - [ ] error_handling strategy
  - [ ] Validation

**Error Handling:**
- [ ] Comprehensive error handling
  - [ ] Corrupted file detection
  - [ ] Unsupported format handling
  - [ ] Clear error messages
  - [ ] Graceful degradation

**Testing:**
- [ ] Unit tests for each extractor
- [ ] Integration tests with real files (PDF, HTML, DOCX)
- [ ] Error handling tests
- [ ] Factory tests

**Deliverable:**
- ✅ Supports .txt, .md, .pdf, .html, .docx
- ✅ Robust error handling
- ✅ Per-file-type configuration
- ✅ All extractors follow same interface (OCP)

### Milestone 5: Configuration, Tools & Polish (Week 5)
**Goal:** Production-ready library with deep configuration

**Deep Configuration:**
- [ ] Complete nested configuration system
  - [ ] All nested config classes fully implemented
  - [ ] `TinyRAGConfig.from_dict()` with nested support
  - [ ] `TinyRAGConfig.load()` from YAML/JSON files
  - [ ] `TinyRAGConfig.save()` to YAML/JSON files
  - [ ] Config merging and inheritance
  - [ ] Preset system ("codebase", "documents", "conversations")
  - [ ] Schema validation (Pydantic or custom)

**Shallow Interface (API):**
- [ ] Ensure API stays simple
  - [ ] Constructor params remain flat (no nested params)
  - [ ] Common params work with defaults
  - [ ] Config object is escape hatch for deep customization
- [ ] Implement convenience methods
  - [ ] `TinyRAG.search()` with query string (not just vector)
  - [ ] `TinyRAG.get_all_chunks()` for context stuffing

**Agent Tools:**
- [ ] Implement `tools/agent_tools.py`
  - [ ] `create_tinyrag()` function
  - [ ] `search_tinyrag()` function
  - [ ] `get_tinyrag_info()` function
  - [ ] Anthropic tool schemas (JSON)
  - [ ] Integration helpers

**Factory Pattern:**
- [ ] Create `factories.py` module
  - [ ] Factory functions for all components
  - [ ] Default implementations
  - [ ] Easy dependency injection

**Error Handling & Validation:**
- [ ] Comprehensive error handling throughout
- [ ] Config validation with clear messages
- [ ] File format validation
- [ ] Graceful error recovery

**Documentation:**
- [ ] Write comprehensive README.md
  - [ ] Quick start (shallow interface)
  - [ ] Deep configuration examples
  - [ ] Architecture overview
- [ ] Write API documentation
  - [ ] All interfaces documented
  - [ ] All classes documented
  - [ ] Usage examples
- [ ] Write .tinyrag format specification
  - [ ] File structure
  - [ ] Schema definitions
  - [ ] Version compatibility

**Testing:**
- [ ] Integration tests for full pipeline
- [ ] Configuration tests (nested, presets, files)
- [ ] Agent tools tests
- [ ] Error handling tests
- [ ] Cross-platform tests (macOS, Linux, Windows)

**PyPI Preparation:**
- [ ] Finalize `setup.py` and `pyproject.toml`
- [ ] Version numbering
- [ ] Package metadata
- [ ] Prepare for publish

**Deliverable:**
- ✅ Deep configuration system complete
- ✅ Shallow API with deep config escape hatch
- ✅ Agent tools integrated
- ✅ Comprehensive documentation
- ✅ Production-ready library
- ✅ Ready for PyPI publish

---

## 7. Success Criteria

### MVP Must-Have

**Functionality:**
- ✅ Create `.tinyrag` file from 5 text files in <10 seconds
- ✅ Load `.tinyrag` file and search in <1 second
- ✅ Accurate retrieval (top-5 chunks are relevant to query)
- ✅ File size reasonable (<10MB for 1000 chunks)
- ✅ Works on macOS, Linux, Windows

**Quality:**
- ✅ Clean, intuitive API
- ✅ Comprehensive error handling
- ✅ Unit test coverage >80%
- ✅ Integration tests for all file formats
- ✅ Documentation (README, API docs, format spec)

**Usability:**
- ✅ Works with defaults (no config needed)
- ✅ Clear error messages
- ✅ Good examples in documentation

### Post-MVP Success Indicators

- Used in at least 3 different projects
- Positive feedback on simplicity
- Agent integrations working
- Community contributions

---

## 8. Scope Control Rules

### Decision Framework

**Before adding any feature, ask:**

1. **Does it align with WHY.md principles?**
   - Portable? ✅
   - Small by design? ✅
   - Retrieval only? ✅
   - Zero infrastructure? ✅

2. **Is it in scope?**
   - Check Section 3 (Scope Boundaries)
   - If not explicitly IN SCOPE, it's OUT OF SCOPE

3. **Does it require infrastructure?**
   - If yes → OUT OF SCOPE
   - If no → Consider

4. **Is it retrieval-only?**
   - If it adds LLM/generation → OUT OF SCOPE
   - If it's retrieval/search → Consider

5. **Is it workspace-scale?**
   - If it's for large-scale → OUT OF SCOPE
   - If it's workspace-scale → Consider

### Change Process

**To add features:**
1. Update this document (FINAL_PLAN.md)
2. Justify against WHY.md principles
3. Update scope boundaries
4. Update implementation phases
5. Get approval

**To remove features:**
1. Update this document
2. Update implementation phases
3. Document reason

---

## 9. What We're NOT Building (Reinforcement)

### Not a Web Service
- ❌ No FastAPI backend
- ❌ No REST API
- ❌ No HTTP endpoints
- ❌ No server deployment

### Not a Full RAG Pipeline
- ❌ No LLM integration
- ❌ No generation
- ❌ No prompt engineering
- ❌ No streaming responses

### Not for Large Scale
- ❌ Not for internet-scale datasets
- ❌ Not for production RAG systems
- ❌ Not for real-time updates
- ❌ Not for high availability

### Not Infrastructure
- ❌ No databases
- ❌ No cloud services
- ❌ No authentication
- ❌ No monitoring

**If you need these things, tinyrag is not the right tool. Use LangChain, Pinecone, or other solutions.**

---

## 10. Project Structure (SOLID Architecture)

```
tinyrag/
├── tinyrag/
│   ├── __init__.py
│   ├── core/                    # Core domain logic
│   │   ├── __init__.py
│   │   ├── rag.py              # TinyRAG orchestrator
│   │   └── chunk.py             # Chunk domain model
│   ├── processing/              # Document processing
│   │   ├── __init__.py
│   │   ├── interfaces.py       # DocumentExtractor, Chunker interfaces
│   │   ├── chunker.py           # Chunker implementation
│   │   └── extractors/          # File type extractors
│   │       ├── __init__.py
│   │       ├── text_extractor.py
│   │       ├── markdown_extractor.py
│   │       ├── pdf_extractor.py
│   │       ├── html_extractor.py
│   │       └── docx_extractor.py
│   ├── embeddings/              # Embedding generation
│   │   ├── __init__.py
│   │   ├── interfaces.py       # EmbeddingGenerator interface
│   │   └── generator.py         # SentenceTransformerGenerator
│   ├── vector_store/            # Vector storage & search
│   │   ├── __init__.py
│   │   ├── interfaces.py       # VectorStore interface
│   │   └── faiss_store.py      # FAISSVectorStore implementation
│   ├── storage/                 # File format & persistence
│   │   ├── __init__.py
│   │   ├── interfaces.py       # RAGSerializer, RAGDeserializer
│   │   ├── serializer.py       # TinyRAGSerializer
│   │   └── deserializer.py     # TinyRAGDeserializer
│   ├── config/                  # Configuration system
│   │   ├── __init__.py
│   │   ├── config.py           # Config classes (nested)
│   │   ├── validators.py       # Config validation
│   │   └── loaders.py          # Config loading (YAML, JSON)
│   ├── tools/                   # Agent tools
│   │   ├── __init__.py
│   │   └── agent_tools.py      # Anthropic tool functions
│   └── factories.py            # Factory functions for dependencies
├── tests/
│   ├── unit/
│   │   ├── test_chunk.py
│   │   ├── test_extractors.py
│   │   ├── test_chunker.py
│   │   ├── test_embeddings.py
│   │   ├── test_vector_store.py
│   │   ├── test_serializer.py
│   │   └── test_config.py
│   └── integration/
│       ├── test_pipeline.py
│       ├── test_file_formats.py
│       └── test_agent_tools.py
├── docs/
│   ├── README.md
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── FORMAT.md
├── setup.py
├── pyproject.toml
└── README.md
```

---

## 11. Dependencies & Constraints

### Hard Constraints

1. **No external services** - Everything must work offline
2. **CPU-friendly** - Must work without GPU
3. **Portable files** - Single file, no dependencies
4. **Python 3.9+** - Support modern Python versions
5. **Open format** - Files must be decodable without library

### Soft Constraints

1. **Small footprint** - Keep dependencies minimal
2. **Fast for small files** - <10 seconds for 5 files
3. **Clear errors** - Helpful error messages
4. **Good defaults** - Works without configuration

---

## 12. Next Steps

### Immediate (Week 1)
1. Set up Python package structure
2. Implement document processor
3. Implement chunker
4. Create basic TinyRAG class

### Follow This Plan
- No deviations without updating this document
- No scope creep
- Focus on MVP
- Iterate based on feedback

---

## 13. Approval & Sign-Off

**This plan is FINAL and APPROVED for implementation.**

**Scope is LOCKED.** Changes require updating this document.

**Key Documents:**
- **WHY.md** - Vision and principles (non-negotiable)
- **ARCHITECTURE.md** - SOLID architecture and component responsibilities
- **DEEP_CONFIG_PROPOSAL.md** - Shallow interface, deep configuration philosophy
- **DECISION_DOCUMENT.md** - Full decision rationale
- **FINAL_PLAN.md** - This document (implementation guide)

---

**Status:** ✅ Ready for Implementation  
**Last Updated:** December 2024  
**Next Review:** After MVP completion
