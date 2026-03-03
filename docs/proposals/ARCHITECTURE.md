# raglet Architecture: SOLID Principles & Clear Responsibilities

## Overview

This document defines the architecture of raglet following SOLID principles to ensure:
- **Single Responsibility** - Each class/module has one clear purpose
- **Open/Closed** - Open for extension, closed for modification
- **Liskov Substitution** - Interfaces can be substituted
- **Interface Segregation** - Small, focused interfaces
- **Dependency Inversion** - Depend on abstractions, not concretions

---

## 1. Architecture Overview

### High-Level Structure

```
raglet/
├── core/                    # Core domain logic
│   ├── rag.py              # RAGlet main class (orchestrator)
│   └── chunk.py             # Chunk domain model
├── processing/              # Document processing
│   ├── extractors/          # File type extractors
│   ├── chunker.py          # Text chunking logic
│   └── interfaces.py       # Processing interfaces
├── embeddings/              # Embedding generation
│   ├── generator.py        # Embedding generator
│   └── interfaces.py       # Embedding interfaces
├── vector_store/            # Vector storage & search
│   ├── faiss_store.py      # FAISS implementation
│   └── interfaces.py       # Vector store interfaces
├── storage/                 # File format & persistence
│   ├── serializer.py       # .raglet serialization
│   ├── deserializer.py     # .raglet deserialization
│   └── interfaces.py       # Storage interfaces
├── config/                  # Configuration system
│   ├── config.py           # Configuration classes
│   ├── validators.py       # Configuration validation
│   └── loaders.py          # Config loading (YAML, JSON)
└── tools/                   # Agent tools
    └── agent_tools.py      # Anthropic tool functions
```

---

## 2. SOLID Principles Applied

### Single Responsibility Principle (SRP)

Each class/module has **one reason to change**.

#### Core Domain (`core/`)

**`RAGlet` (orchestrator)**
- **Responsibility:** Orchestrate the RAG pipeline
- **Changes when:** Pipeline flow changes
- **Does NOT:** Process documents, generate embeddings, or search

**`Chunk` (domain model)**
- **Responsibility:** Represent a text chunk with metadata
- **Changes when:** Chunk structure changes
- **Does NOT:** Process text, search, or serialize

#### Document Processing (`processing/`)

**`DocumentExtractor` (interface)**
- **Responsibility:** Define contract for extracting text from files
- **Changes when:** Extraction contract changes

**`TextExtractor`, `PDFExtractor`, `HTMLExtractor` (implementations)**
- **Responsibility:** Extract text from specific file types
- **Changes when:** File format parsing changes

**`Chunker`**
- **Responsibility:** Split text into chunks
- **Changes when:** Chunking strategy changes
- **Does NOT:** Extract text or generate embeddings

#### Embeddings (`embeddings/`)

**`EmbeddingGenerator` (interface)**
- **Responsibility:** Define contract for generating embeddings
- **Changes when:** Embedding contract changes

**`SentenceTransformerGenerator` (implementation)**
- **Responsibility:** Generate embeddings using sentence-transformers
- **Changes when:** Embedding model or library changes
- **Does NOT:** Process documents or search

#### Vector Store (`vector_store/`)

**`VectorStore` (interface)**
- **Responsibility:** Define contract for vector storage and search
- **Changes when:** Vector store contract changes

**`FAISSVectorStore` (implementation)**
- **Responsibility:** Store vectors and search using FAISS
- **Changes when:** FAISS usage or index type changes
- **Does NOT:** Generate embeddings or serialize

#### Storage (`storage/`)

**`RAGSerializer` (interface)**
- **Responsibility:** Define contract for serializing RAG to file
- **Changes when:** Serialization contract changes

**`RAGletSerializer` (implementation)**
- **Responsibility:** Serialize RAGlet to .raglet format
- **Changes when:** File format changes
- **Does NOT:** Process documents or search

**`RAGDeserializer` (interface)**
- **Responsibility:** Define contract for deserializing RAG from file
- **Changes when:** Deserialization contract changes

**`RAGletDeserializer` (implementation)**
- **Responsibility:** Deserialize .raglet file to RAGlet
- **Changes when:** File format changes
- **Does NOT:** Process documents or search

#### Configuration (`config/`)

**`RAGletConfig`**
- **Responsibility:** Hold configuration data
- **Changes when:** Configuration structure changes
- **Does NOT:** Validate or load configs

**`ConfigValidator`**
- **Responsibility:** Validate configuration values
- **Changes when:** Validation rules change
- **Does NOT:** Hold or load configs

**`ConfigLoader`**
- **Responsibility:** Load configuration from files
- **Changes when:** File format or loading logic changes
- **Does NOT:** Validate or hold configs

---

### Open/Closed Principle (OCP)

**Open for extension, closed for modification.**

#### Example: Adding New File Type

**Current Design:**
```python
# Interface (closed for modification)
class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> str:
        pass

# Implementation (open for extension)
class PDFExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> str:
        # PDF extraction logic
        pass

# New file type (extension, not modification)
class DOCXExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> str:
        # DOCX extraction logic
        pass
```

**No modification needed** - just add new extractor class.

#### Example: Adding New Embedding Model

**Current Design:**
```python
# Interface (closed for modification)
class EmbeddingGenerator(ABC):
    @abstractmethod
    def generate(self, texts: List[str]) -> np.ndarray:
        pass

# Implementation (open for extension)
class SentenceTransformerGenerator(EmbeddingGenerator):
    def generate(self, texts: List[str]) -> np.ndarray:
        # sentence-transformers logic
        pass

# New model (extension, not modification)
class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    def generate(self, texts: List[str]) -> np.ndarray:
        # OpenAI API logic
        pass
```

**No modification needed** - just add new generator class.

---

### Liskov Substitution Principle (LSP)

**Subtypes must be substitutable for their base types.**

#### Example: Vector Store Implementations

```python
# Base interface
class VectorStore(ABC):
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        pass

# FAISS implementation (substitutable)
class FAISSVectorStore(VectorStore):
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        # FAISS implementation
        pass
    
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        # FAISS implementation
        pass

# ChromaDB implementation (substitutable)
class ChromaDBVectorStore(VectorStore):
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        # ChromaDB implementation
        pass
    
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        # ChromaDB implementation
        pass
```

**Any VectorStore implementation can be substituted** without changing client code.

---

### Interface Segregation Principle (ISP)

**Clients should not depend on interfaces they don't use.**

#### Example: Separate Read/Write Interfaces

**Bad Design (violates ISP):**
```python
class VectorStore(ABC):
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> None:
        pass
    
    @abstractmethod
    def update_vectors(self, ids: List[str], vectors: np.ndarray) -> None:
        pass
```

**Problem:** Read-only clients forced to implement write methods.

**Good Design (follows ISP):**
```python
# Separate interfaces
class VectorStoreReader(ABC):
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        pass

class VectorStoreWriter(ABC):
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        pass

# Combined interface (for implementations that support both)
class VectorStore(VectorStoreReader, VectorStoreWriter):
    pass
```

**Clients depend only on interfaces they use.**

---

### Dependency Inversion Principle (DIP)

**Depend on abstractions, not concretions.**

#### Example: RAGlet Depends on Interfaces

**Bad Design (violates DIP):**
```python
class RAGlet:
    def __init__(self):
        self.chunker = Chunker()  # Depends on concrete class
        self.embedder = SentenceTransformerGenerator()  # Depends on concrete class
        self.vector_store = FAISSVectorStore()  # Depends on concrete class
```

**Problem:** Tightly coupled to specific implementations.

**Good Design (follows DIP):**
```python
class RAGlet:
    def __init__(
        self,
        chunker: Chunker,
        embedding_generator: EmbeddingGenerator,  # Interface
        vector_store: VectorStore,  # Interface
        config: RAGletConfig
    ):
        self.chunker = chunker
        self.embedding_generator = embedding_generator  # Abstraction
        self.vector_store = vector_store  # Abstraction
        self.config = config
```

**Depends on abstractions** - can swap implementations easily.

---

## 3. Component Responsibilities

### Core Domain (`core/`)

#### `RAGlet` (Orchestrator)

**Responsibilities:**
- Orchestrate RAG pipeline (process → chunk → embed → index)
- Provide public API (from_files, save, load, search)
- Coordinate between components
- Manage lifecycle

**Dependencies:**
- `Chunker` (interface)
- `EmbeddingGenerator` (interface)
- `VectorStore` (interface)
- `RAGSerializer` (interface)
- `RAGDeserializer` (interface)
- `RAGletConfig`

**Does NOT:**
- Process documents directly
- Generate embeddings directly
- Search vectors directly
- Serialize/deserialize directly

#### `Chunk` (Domain Model)

**Responsibilities:**
- Represent a text chunk
- Hold chunk metadata
- Provide accessors

**Dependencies:**
- None (pure data class)

**Does NOT:**
- Process text
- Search
- Serialize

---

### Document Processing (`processing/`)

#### `DocumentExtractor` (Interface)

**Responsibilities:**
- Define contract for text extraction
- Specify input/output format

**Dependencies:**
- None (abstract)

#### `TextExtractor`, `PDFExtractor`, etc. (Implementations)

**Responsibilities:**
- Extract text from specific file types
- Handle file format errors
- Return extracted text

**Dependencies:**
- File I/O libraries (PyPDF2, BeautifulSoup, etc.)

**Does NOT:**
- Chunk text
- Generate embeddings
- Store vectors

#### `Chunker`

**Responsibilities:**
- Split text into chunks
- Apply chunking strategy
- Preserve metadata

**Dependencies:**
- `RAGletConfig` (for chunking config)

**Does NOT:**
- Extract text
- Generate embeddings
- Store vectors

---

### Embeddings (`embeddings/`)

#### `EmbeddingGenerator` (Interface)

**Responsibilities:**
- Define contract for embedding generation
- Specify input/output format

**Dependencies:**
- None (abstract)

#### `SentenceTransformerGenerator` (Implementation)

**Responsibilities:**
- Generate embeddings using sentence-transformers
- Handle batching
- Manage model loading

**Dependencies:**
- `sentence-transformers` library
- `RAGletConfig` (for embedding config)

**Does NOT:**
- Process documents
- Store vectors
- Search vectors

---

### Vector Store (`vector_store/`)

#### `VectorStore` (Interface)

**Responsibilities:**
- Define contract for vector storage and search
- Specify input/output format

**Dependencies:**
- None (abstract)

#### `FAISSVectorStore` (Implementation)

**Responsibilities:**
- Store vectors in FAISS index
- Search vectors using FAISS
- Manage index lifecycle

**Dependencies:**
- `faiss` library
- `numpy`

**Does NOT:**
- Generate embeddings
- Process documents
- Serialize/deserialize

---

### Storage (`storage/`)

#### `RAGSerializer` (Interface)

**Responsibilities:**
- Define contract for serialization
- Specify output format

**Dependencies:**
- None (abstract)

#### `RAGletSerializer` (Implementation)

**Responsibilities:**
- Serialize RAGlet to .raglet format
- Handle compression
- Write to file

**Dependencies:**
- `zipfile` (compression)
- `json` (metadata)
- `numpy` (embeddings)
- `faiss` (index serialization)

**Does NOT:**
- Process documents
- Generate embeddings
- Search vectors

#### `RAGDeserializer` (Interface)

**Responsibilities:**
- Define contract for deserialization
- Specify input format

**Dependencies:**
- None (abstract)

#### `RAGletDeserializer` (Implementation)

**Responsibilities:**
- Deserialize .raglet file to RAGlet
- Handle decompression
- Read from file

**Dependencies:**
- `zipfile` (decompression)
- `json` (metadata)
- `numpy` (embeddings)
- `faiss` (index deserialization)

**Does NOT:**
- Process documents
- Generate embeddings
- Search vectors

---

### Configuration (`config/`)

#### `RAGletConfig`

**Responsibilities:**
- Hold configuration data
- Provide accessors
- Support merging

**Dependencies:**
- Nested config classes (`ChunkingConfig`, etc.)

**Does NOT:**
- Validate (delegates to `ConfigValidator`)
- Load (delegates to `ConfigLoader`)

#### `ConfigValidator`

**Responsibilities:**
- Validate configuration values
- Check constraints
- Raise validation errors

**Dependencies:**
- `RAGletConfig`

**Does NOT:**
- Hold config data
- Load configs

#### `ConfigLoader`

**Responsibilities:**
- Load configuration from files (YAML, JSON)
- Parse configuration data
- Return `RAGletConfig` instance

**Dependencies:**
- `yaml` or `json` libraries
- `RAGletConfig`

**Does NOT:**
- Validate configs
- Hold config data

---

## 4. Dependency Graph

```
RAGlet (orchestrator)
    ├── Chunker
    │   └── RAGletConfig
    ├── DocumentExtractor (interface)
    │   ├── TextExtractor
    │   ├── PDFExtractor
    │   └── HTMLExtractor
    ├── EmbeddingGenerator (interface)
    │   └── SentenceTransformerGenerator
    │       └── RAGletConfig
    ├── VectorStore (interface)
    │   └── FAISSVectorStore
    ├── RAGSerializer (interface)
    │   └── RAGletSerializer
    ├── RAGDeserializer (interface)
    │   └── RAGletDeserializer
    └── RAGletConfig
        ├── ChunkingConfig
        ├── EmbeddingConfig
        ├── SearchConfig
        └── FileProcessingConfig
```

**Key Points:**
- RAGlet depends on interfaces, not implementations
- Components depend on config, not each other
- Clear separation of concerns

---

## 5. Interface Definitions

### Document Processing Interfaces

```python
# processing/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DocumentExtractor(ABC):
    """Interface for extracting text from files."""
    
    @abstractmethod
    def extract(self, file_path: str) -> str:
        """Extract text from file."""
        pass
    
    @abstractmethod
    def can_extract(self, file_path: str) -> bool:
        """Check if this extractor can handle the file."""
        pass

class Chunker(ABC):
    """Interface for chunking text."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List["Chunk"]:
        """Split text into chunks."""
        pass
```

### Embedding Interfaces

```python
# embeddings/interfaces.py

from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingGenerator(ABC):
    """Interface for generating embeddings."""
    
    @abstractmethod
    def generate(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass
```

### Vector Store Interfaces

```python
# vector_store/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class VectorStoreReader(ABC):
    """Interface for reading/searching vectors."""
    
    @abstractmethod
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

class VectorStoreWriter(ABC):
    """Interface for writing vectors."""
    
    @abstractmethod
    def add_vectors(
        self, 
        vectors: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add vectors to store."""
        pass

class VectorStore(VectorStoreReader, VectorStoreWriter):
    """Combined interface for read/write operations."""
    pass
```

### Storage Interfaces

```python
# storage/interfaces.py

from abc import ABC, abstractmethod
from typing import BinaryIO

class RAGSerializer(ABC):
    """Interface for serializing RAG to file."""
    
    @abstractmethod
    def serialize(self, rag: "RAGlet", file_path: str) -> None:
        """Serialize RAG to file."""
        pass

class RAGDeserializer(ABC):
    """Interface for deserializing RAG from file."""
    
    @abstractmethod
    def deserialize(self, file_path: str) -> "RAGlet":
        """Deserialize RAG from file."""
        pass
```

---

## 6. Implementation Example

### RAGlet Class (Following SOLID)

```python
# core/rag.py

from typing import List, Optional
from .chunk import Chunk
from processing.interfaces import DocumentExtractor, Chunker
from embeddings.interfaces import EmbeddingGenerator
from vector_store.interfaces import VectorStore
from storage.interfaces import RAGSerializer, RAGDeserializer
from config.config import RAGletConfig

class RAGlet:
    """
    Main RAG class - orchestrates the pipeline.
    
    Follows Dependency Inversion: depends on interfaces, not implementations.
    Follows Single Responsibility: orchestrates, doesn't implement.
    """
    
    def __init__(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        vector_store: VectorStore,  # Interface, not concrete
        config: RAGletConfig
    ):
        self.chunks = chunks
        self.embeddings = embeddings
        self.vector_store = vector_store  # Abstraction
        self.config = config
    
    @classmethod
    def from_files(
        cls,
        files: List[str],
        document_extractor: DocumentExtractor,  # Interface
        chunker: Chunker,  # Interface
        embedding_generator: EmbeddingGenerator,  # Interface
        vector_store_factory: callable,  # Factory for VectorStore
        config: RAGletConfig
    ) -> "RAGlet":
        """
        Create RAGlet from files.
        
        Depends on interfaces, not concrete implementations.
        """
        # Extract text
        texts = []
        for file_path in files:
            text = document_extractor.extract(file_path)
            texts.append(text)
        
        # Chunk text
        all_chunks = []
        for text in texts:
            chunks = chunker.chunk(text, metadata={"source": file_path})
            all_chunks.extend(chunks)
        
        # Generate embeddings
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = embedding_generator.generate(chunk_texts)
        
        # Create vector store
        vector_store = vector_store_factory(embeddings.shape[1])
        vector_store.add_vectors(embeddings, [chunk.to_dict() for chunk in all_chunks])
        
        return cls(
            chunks=all_chunks,
            embeddings=embeddings,
            vector_store=vector_store,
            config=config
        )
    
    def search(
        self,
        query: str,
        embedding_generator: EmbeddingGenerator,  # Interface
        top_k: Optional[int] = None
    ) -> List[Chunk]:
        """Search for relevant chunks."""
        top_k = top_k or self.config.search.default_top_k
        
        # Generate query embedding
        query_vector = embedding_generator.generate([query])[0]
        
        # Search vector store
        results = self.vector_store.search(
            query_vector,
            top_k=top_k,
            score_threshold=self.config.search.similarity_threshold
        )
        
        # Map back to chunks
        return [self.chunks[r["index"]] for r in results]
    
    def save(
        self,
        file_path: str,
        serializer: RAGSerializer  # Interface
    ) -> None:
        """Save to .raglet file."""
        serializer.serialize(self, file_path)
    
    @classmethod
    def load(
        cls,
        file_path: str,
        deserializer: RAGDeserializer,  # Interface
        embedding_generator: EmbeddingGenerator,  # Interface
        vector_store_factory: callable  # Factory
    ) -> "RAGlet":
        """Load from .raglet file."""
        return deserializer.deserialize(file_path, embedding_generator, vector_store_factory)
```

---

## 7. Factory Pattern for Dependencies

### Factory Functions

```python
# factories.py

from processing.extractors import TextExtractor, PDFExtractor, HTMLExtractor
from processing.chunker import Chunker
from embeddings.generator import SentenceTransformerGenerator
from vector_store.faiss_store import FAISSVectorStore
from storage.serializer import RAGletSerializer
from storage.deserializer import RAGletDeserializer
from config.config import RAGletConfig

def create_document_extractor(file_path: str) -> DocumentExtractor:
    """Factory for document extractors."""
    if file_path.endswith('.pdf'):
        return PDFExtractor()
    elif file_path.endswith('.html'):
        return HTMLExtractor()
    else:
        return TextExtractor()

def create_chunker(config: RAGletConfig) -> Chunker:
    """Factory for chunker."""
    return Chunker(config.chunking)

def create_embedding_generator(config: RAGletConfig) -> EmbeddingGenerator:
    """Factory for embedding generator."""
    return SentenceTransformerGenerator(config.embeddings)

def create_vector_store(dimension: int) -> VectorStore:
    """Factory for vector store."""
    return FAISSVectorStore(dimension)

def create_serializer() -> RAGSerializer:
    """Factory for serializer."""
    return RAGletSerializer()

def create_deserializer() -> RAGDeserializer:
    """Factory for deserializer."""
    return RAGletDeserializer()
```

### Usage with Factories

```python
# Simple usage - factories handle dependencies
config = RAGletConfig()
rag = RAGlet.from_files(
    files=["doc.txt"],
    document_extractor=create_document_extractor("doc.txt"),
    chunker=create_chunker(config),
    embedding_generator=create_embedding_generator(config),
    vector_store_factory=create_vector_store,
    config=config
)
```

---

## 8. Benefits of This Architecture

### Testability
- **Mock interfaces** - Easy to mock dependencies
- **Isolated tests** - Test components independently
- **No side effects** - Pure functions where possible

### Maintainability
- **Clear responsibilities** - Each class has one job
- **Easy to change** - Modify one component without affecting others
- **Self-documenting** - Structure shows responsibilities

### Extensibility
- **Add new file types** - Implement `DocumentExtractor`
- **Add new embedding models** - Implement `EmbeddingGenerator`
- **Add new vector stores** - Implement `VectorStore`
- **No modification** - Extend, don't modify

### Flexibility
- **Swap implementations** - Change concrete classes easily
- **Configuration-driven** - Behavior controlled by config
- **Composable** - Mix and match components

---

## 9. Summary

### SOLID Principles Applied

1. **Single Responsibility** ✅
   - Each class has one clear purpose
   - Changes isolated to one component

2. **Open/Closed** ✅
   - Open for extension (new extractors, generators, stores)
   - Closed for modification (interfaces stable)

3. **Liskov Substitution** ✅
   - Implementations substitutable
   - Interfaces define contracts

4. **Interface Segregation** ✅
   - Small, focused interfaces
   - Clients depend only on what they use

5. **Dependency Inversion** ✅
   - Depend on abstractions (interfaces)
   - Not concretions (implementations)

### Clear Responsibilities

- **RAGlet** - Orchestrates pipeline
- **Extractors** - Extract text from files
- **Chunker** - Split text into chunks
- **EmbeddingGenerator** - Generate embeddings
- **VectorStore** - Store and search vectors
- **Serializer/Deserializer** - Persist to file
- **Config** - Hold configuration

### Architecture Benefits

- **Testable** - Easy to test components
- **Maintainable** - Clear structure
- **Extensible** - Easy to add features
- **Flexible** - Swap implementations

---

**This architecture ensures clarity, maintainability, and extensibility while following SOLID principles.**
