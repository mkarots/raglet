# CLAUDE.md - Agent Instructions for raglet

**This file contains instructions for AI agents working on the raglet codebase.**

---

## 1. Project Overview

### What is raglet?

**raglet** is a Python library that creates portable `.raglet` files - single-file knowledge bases containing chunks, embeddings, and metadata. It's retrieval-only (no LLM), zero infrastructure, and designed for workspace-scale problems.

### Core Principles (NON-NEGOTIABLE)

1. **Portable** - One `.raglet` file. Save it, git commit it, email it
2. **Small by design** - Workspace-scale (codebases, conversations, notes). **Not the internet**
3. **Retrieval only** - raglet finds chunks. You decide what to do with them. **Bring your own LLM**
4. **Open format** - The `.raglet` file is easily decodable. Embeddings are extractable. No lock-in
5. **Zero infrastructure** - `pip install raglet`. That's it

### Key Documents

- **`docs/problems/00-problem-statement.md`** - Why raglet exists
- **`docs/decisions/`** - All architectural decisions (ADR format)
- **`plans/FINAL_PLAN.md`** - Implementation plan and roadmap
- **`docs/proposals/ARCHITECTURE.md`** - SOLID architecture details

---

## 2. Architecture & Design Principles

### SOLID Principles (MANDATORY)

**Every component MUST follow SOLID principles:**

1. **Single Responsibility** - Each class/module has ONE clear purpose
2. **Open/Closed** - Open for extension, closed for modification (use interfaces)
3. **Liskov Substitution** - Implementations must be substitutable
4. **Interface Segregation** - Small, focused interfaces
5. **Dependency Inversion** - Depend on abstractions (interfaces), not concretions

### Component Structure

```
raglet/
├── core/                    # Core domain logic
│   ├── rag.py              # RAGlet orchestrator (depends on interfaces)
│   └── chunk.py             # Chunk domain model
├── processing/              # Document processing
│   ├── interfaces.py       # DocumentExtractor, Chunker interfaces
│   ├── chunker.py          # Chunker implementation
│   └── extractors/          # File type extractors (implement DocumentExtractor)
├── embeddings/              # Embedding generation
│   ├── interfaces.py       # EmbeddingGenerator interface
│   └── generator.py         # SentenceTransformerGenerator implementation
├── vector_store/            # Vector storage & search
│   ├── interfaces.py       # VectorStore interface
│   └── faiss_store.py      # FAISSVectorStore implementation
├── storage/                 # File format & persistence
│   ├── interfaces.py       # RAGSerializer, RAGDeserializer interfaces
│   ├── serializer.py       # RAGletSerializer implementation
│   └── deserializer.py     # RAGletDeserializer implementation
├── config/                  # Configuration system
│   ├── config.py           # Config classes (nested structure)
│   ├── validators.py       # Config validation
│   └── loaders.py          # Config loading (YAML, JSON)
└── tools/                   # Agent tools
    └── agent_tools.py      # Anthropic tool functions
```

### Key Rules

1. **RAGlet depends on interfaces, NOT concrete classes**
2. **Each component has ONE responsibility**
3. **New implementations extend interfaces, don't modify them**
4. **Configuration is deep, API is shallow**

---

## 3. Coding Conventions

### Python Style

- **Python 3.9+** compatibility required
- **Type hints** - Use type hints for all function signatures
- **Docstrings** - Google-style docstrings for all public methods
- **Line length** - 100 characters (configured in pyproject.toml)

### Code Formatting

- **Black** - Code formatting (100 char line length)
- **Ruff** - Linting (configured in pyproject.toml)
- **MyPy** - Type checking (optional strict mode)

### Naming Conventions

- **Classes** - PascalCase: `RAGlet`, `ChunkingConfig`
- **Functions/Methods** - snake_case: `from_files()`, `get_all_chunks()`
- **Interfaces** - PascalCase with descriptive names: `DocumentExtractor`, `EmbeddingGenerator`
- **Implementations** - Descriptive names: `SentenceTransformerGenerator`, `FAISSVectorStore`

### Import Organization

```python
# Standard library
import os
from typing import List, Optional

# Third-party
import numpy as np

# Local imports
from raglet.core.chunk import Chunk
from raglet.processing.interfaces import DocumentExtractor
```

### Error Handling

- **Raise specific exceptions** - `FileNotFoundError`, `ValueError`, not generic `Exception`
- **Clear error messages** - Explain what went wrong and how to fix it
- **Validate early** - Check inputs at function boundaries

### Example: Adding a New Extractor

```python
# processing/extractors/yaml_extractor.py

from raglet.processing.interfaces import DocumentExtractor

class YAMLExtractor(DocumentExtractor):
    """Extracts text from YAML files."""
    
    def can_extract(self, file_path: str) -> bool:
        """Check if file is a YAML file."""
        return file_path.lower().endswith(('.yaml', '.yml'))
    
    def extract(self, file_path: str) -> str:
        """Extract text from YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Extracted text content (raw YAML)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is invalid
        """
        # Implementation here
        pass
```

**Key points:**
- Implements `DocumentExtractor` interface
- Single responsibility: extract YAML text
- Clear docstring with Args/Returns/Raises
- Type hints on all methods
- Note: raglet focuses on text files only (no binary formats)

---

## 4. Testing Requirements

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests (component interactions)
└── e2e/              # End-to-end tests (full pipeline)
```

### Test Markers

Use pytest markers:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests

### Test Requirements

1. **Unit tests** - Test each component in isolation
2. **Integration tests** - Test component interactions
3. **E2E tests** - Test full pipeline
4. **Mock interfaces** - Use mocks for dependencies (follows DIP)
5. **Coverage** - Aim for >80% coverage

### Example: Unit Test

```python
# tests/unit/test_pdf_extractor.py

import pytest
from raglet.processing.extractors.pdf_extractor import PDFExtractor

@pytest.mark.unit
class TestPDFExtractor:
    """Test PDFExtractor."""
    
    def test_can_extract_pdf(self):
        """Test can_extract for PDF files."""
        extractor = PDFExtractor()
        assert extractor.can_extract("test.pdf") is True
    
    def test_cannot_extract_txt(self):
        """Test can_extract returns False for non-PDF files."""
        extractor = PDFExtractor()
        assert extractor.can_extract("test.txt") is False
```

### Running Tests

```bash
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-e2e         # E2E tests only
make test             # All tests
```

---

## 5. Documentation Structure

### Documentation Organization

All documentation lives in `docs/` directory:

```
docs/
├── decisions/        # Architectural decisions (ADR format)
├── proposals/        # Design proposals and explorations
├── problems/         # Problem statements
└── README.md         # Documentation index
```

### Writing Documentation

#### Decisions (`docs/decisions/`)

**When to write:** When making architectural or design decisions

**Format:** ADR (Architecture Decision Record)

**File naming:** `XXX-decision-name.md` (sequential numbers)

**Template:**
```markdown
# Decision XXX: Decision Name

**Date:** YYYY-MM-DD  
**Status:** Accepted/Rejected/Proposed

## Context

What problem does this solve?

## Decision

What did we decide?

## Rationale

Why did we decide this?

## Consequences

What does this mean? What are the trade-offs?

## Alternatives Considered

What other options did we consider?
```

**Example:** See `docs/decisions/001-library-not-service.md`

#### Proposals (`docs/proposals/`)

**When to write:** When exploring design options before making decisions

**Format:** Free-form, but should include:
- Problem statement
- Options explored
- Recommendation
- Implementation plan

**File naming:** `PROPOSAL_NAME.md` (descriptive)

**Example:** See `docs/proposals/DEEP_CONFIG_PROPOSAL.md`

#### Problems (`docs/problems/`)

**When to write:** When documenting a problem or use case

**Format:** Problem statement with context

**File naming:** `XX-problem-name.md` (sequential numbers)

**Example:** See `docs/problems/00-problem-statement.md`

### Documentation Rules

1. **Update README.md** - When adding new docs, update the relevant README
2. **Link between docs** - Cross-reference related documents
3. **Keep it current** - Update docs when code changes
4. **Be specific** - Include examples and code snippets

---

## 6. Configuration Philosophy

### Shallow Interface, Deep Configuration

**API stays simple:**
```python
# Simple - just works
rag = RAGlet.from_files(["doc.txt"])

# Override common params - still simple
rag = RAGlet.from_files(["doc.txt"], chunk_size=1024)
```

**Configuration goes deep:**
```python
# Deep customization via config
config = RAGletConfig(
    chunking=ChunkingConfig(size=1024, strategy="semantic"),
    embeddings=EmbeddingConfig(model="all-mpnet-base-v2")
)
rag = RAGlet.from_files(["doc.txt"], config=config)
```

### Rules

1. **No nested parameters in constructor** - Keep API shallow
2. **Configuration is escape hatch** - Complex options live in config
3. **Progressive disclosure** - Simple → Override → Preset → Deep config

---

## 7. Adding New Features

### Decision Framework

**Before adding ANY feature, ask:**

1. **Does it align with WHY.md principles?**
   - Portable? ✅
   - Small by design? ✅
   - Retrieval only? ✅
   - Zero infrastructure? ✅

2. **Is it in scope?**
   - Check `plans/FINAL_PLAN.md` Section 3 (Scope Boundaries)
   - If not explicitly IN SCOPE, it's OUT OF SCOPE

3. **Does it require infrastructure?**
   - If yes → OUT OF SCOPE

4. **Is it retrieval-only?**
   - If it adds LLM/generation → OUT OF SCOPE

5. **Is it workspace-scale?**
   - If it's for large-scale → OUT OF SCOPE

### Process

1. **Check scope** - Is it in FINAL_PLAN.md?
2. **Check decisions** - Review relevant decisions in `docs/decisions/`
3. **Follow SOLID** - Design with interfaces, single responsibility
4. **Write tests** - Unit, integration, and E2E tests
5. **Update docs** - Add/update documentation
6. **Update FINAL_PLAN.md** - If scope changes

### Example: Adding a New Text File Format Extractor

1. ✅ Check scope - Text file formats are IN SCOPE (FINAL_PLAN.md)
2. ✅ Check decisions - Follows Decision 002 (SOLID Architecture)
3. ✅ Implement extractor - Implements `DocumentExtractor` interface
4. ✅ Write tests - Unit tests for the extractor
5. ✅ Update factory - Add extractor to factory
6. ✅ Update docs - Document the new format support

---

## 8. Code Review Checklist

When reviewing or writing code, ensure:

- [ ] Follows SOLID principles
- [ ] Implements interfaces correctly
- [ ] Has type hints
- [ ] Has docstrings
- [ ] Has unit tests
- [ ] Has integration tests (if applicable)
- [ ] Error handling is appropriate
- [ ] No hardcoded values (use config)
- [ ] Follows naming conventions
- [ ] No circular dependencies
- [ ] Dependencies are injected (not created internally)

---

## 9. Common Patterns

### Factory Pattern

Use factories to create implementations:

```python
# factories.py
def create_extractor(file_path: str) -> DocumentExtractor:
    """Factory for document extractors."""
    if file_path.endswith('.md'):
        return MarkdownExtractor()
    elif file_path.endswith('.txt'):
        return TextExtractor()
    else:
        # Fallback to TextExtractor for all other text files
        # (source code, config files, etc.)
        return TextExtractor()
```

### Dependency Injection

RAGlet receives dependencies, doesn't create them:

```python
# Good (Dependency Inversion)
def __init__(
    self,
    embedding_generator: EmbeddingGenerator,  # Interface
    vector_store: VectorStore  # Interface
):
    self.embedding_generator = embedding_generator
    self.vector_store = vector_store

# Bad (violates DIP)
def __init__(self):
    self.embedding_generator = SentenceTransformerGenerator()  # Concrete
    self.vector_store = FAISSVectorStore()  # Concrete
```

### Configuration Validation

Always validate configuration:

```python
def validate(self) -> None:
    """Validate configuration."""
    if self.size < 1:
        raise ValueError("chunk_size must be >= 1")
    if self.overlap >= self.size:
        raise ValueError("chunk_overlap must be < chunk_size")
```

---

## 10. Testing Patterns

### Mocking Interfaces

```python
from unittest.mock import Mock
from raglet.processing.interfaces import DocumentExtractor

def test_raglet_with_mock_extractor():
    """Test RAGlet with mocked extractor."""
    mock_extractor = Mock(spec=DocumentExtractor)
    mock_extractor.extract.return_value = "Test text"
    
    # Use mock in test
    rag = RAGlet.from_files(["test.txt"], document_extractor=mock_extractor)
```

### Integration Test Pattern

```python
@pytest.mark.integration
def test_extract_chunk_flow():
    """Test extract → chunk integration."""
    # Use real implementations
    extractor = TextExtractor()
    chunker = SentenceAwareChunker(ChunkingConfig())
    
    text = extractor.extract("test.txt")
    chunks = chunker.chunk(text, {})
    
    assert len(chunks) > 0
```

### E2E Test Pattern

```python
@pytest.mark.e2e
def test_e2e_create_from_files():
    """E2E test: Create RAGlet from files."""
    # Test full pipeline
    rag = RAGlet.from_files(["doc1.txt", "doc2.md"])
    
    # Verify results
    assert len(rag.chunks) > 0
    assert all(chunk.text for chunk in rag.chunks)
```

---

## 11. Documentation Writing Guide

### When to Write Documentation

- **Decisions** - When making architectural/design decisions
- **Proposals** - When exploring design options
- **Problems** - When documenting a problem or use case
- **Code** - Always document public APIs

### Documentation Location

- **Code docs** - In docstrings (Google style)
- **Architecture docs** - `docs/proposals/ARCHITECTURE.md`
- **Decisions** - `docs/decisions/XXX-decision-name.md`
- **Proposals** - `docs/proposals/PROPOSAL_NAME.md`
- **Problems** - `docs/problems/XX-problem-name.md`

### Documentation Format

**Code Docstrings (Google style):**
```python
def extract(self, file_path: str) -> str:
    """Extract text from file.
    
    Args:
        file_path: Path to the file to extract text from
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    pass
```

**Decision Docs:**
- Use ADR format (see `docs/decisions/README.md`)
- Include context, decision, rationale, consequences
- Reference related decisions

**Proposal Docs:**
- Problem statement
- Options explored
- Recommendation
- Implementation plan

---

## 12. CI/CD Requirements

### Makefile Commands

All code must work with:

```bash
make install-dev     # Install with dev dependencies (uses uv)
make lint           # Run linters (must pass)
make format         # Format code (must pass)
make type-check     # Type checking (should pass)
make test           # All tests (must pass)
make test-unit      # Unit tests
make test-integration # Integration tests
make test-e2e       # E2E tests
make ci             # Full CI pipeline
```

**Note:** The project uses [uv](https://github.com/astral-sh/uv) for fast package management. Install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`. The Makefile uses `uv` for all package operations. Run `make help` to verify uv is detected.

### GitHub Actions

- Runs on push/PR to main/develop
- Tests on multiple OS (Ubuntu, macOS, Windows)
- Tests on Python 3.9-3.12
- Runs linting, formatting, type checking
- Generates coverage reports

**All checks must pass before merging.**

---

## 13. Scope Boundaries

### ✅ IN SCOPE

- Python library (not web service)
- Portable `.raglet` files
- Document processing: Text files only (.txt, .md, source code, config files, etc.)
- Local embeddings (sentence-transformers)
- FAISS vector search
- Configuration system
- Agent tools

### ❌ OUT OF SCOPE

- Web service/API
- LLM integration
- Real-time updates
- Large-scale datasets
- Database backends
- Cloud services
- Authentication
- Monitoring

**If unsure, check `plans/FINAL_PLAN.md` Section 3.**

---

## 14. Quick Reference

### Project Structure
- **Core logic:** `raglet/core/`
- **Processing:** `raglet/processing/`
- **Embeddings:** `raglet/embeddings/`
- **Vector store:** `raglet/vector_store/`
- **Storage:** `raglet/storage/`
- **Config:** `raglet/config/`
- **Tools:** `raglet/tools/`

### Key Files
- **Plan:** `plans/FINAL_PLAN.md`
- **Architecture:** `docs/proposals/ARCHITECTURE.md`
- **Decisions:** `docs/decisions/`
- **Problems:** `docs/problems/`

### Key Principles
1. SOLID architecture
2. Shallow interface, deep configuration
3. Portable files
4. Zero infrastructure
5. Retrieval only

### Testing
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- E2E tests: `tests/e2e/`
- Run: `make test`

### Documentation
- Decisions: `docs/decisions/XXX-decision-name.md`
- Proposals: `docs/proposals/PROPOSAL_NAME.md`
- Problems: `docs/problems/XX-problem-name.md`

---

## 15. Agent Workflow

### When Starting Work

1. **Read WHY.md** - Understand the vision
2. **Read FINAL_PLAN.md** - Understand scope
3. **Check decisions/** - Review relevant decisions
4. **Check architecture** - Understand SOLID structure

### When Writing Code

1. **Follow SOLID** - Single responsibility, interfaces, dependency injection
2. **Write tests** - Unit, integration, E2E as appropriate
3. **Type hints** - All function signatures
4. **Docstrings** - Google style for public APIs
5. **Run tests** - `make test` before committing

### When Making Decisions

1. **Check scope** - Is it in FINAL_PLAN.md?
2. **Write decision doc** - `docs/decisions/XXX-decision-name.md`
3. **Update README** - Add to `docs/decisions/README.md`
4. **Update FINAL_PLAN.md** - If scope changes

### When Writing Documentation

1. **Choose location** - decisions/, proposals/, or problems/
2. **Follow format** - ADR for decisions, free-form for proposals
3. **Update README** - Add to relevant README.md
4. **Cross-reference** - Link related documents

---

## 16. Common Mistakes to Avoid

### ❌ Don't Do This

1. **Violate SOLID**
   ```python
   # Bad: RAGlet creates concrete classes
   def __init__(self):
       self.embedder = SentenceTransformerGenerator()
   ```

2. **Add infrastructure**
   ```python
   # Bad: Requires external service
   def generate_embeddings(self):
       return openai.Embedding.create(...)
   ```

3. **Add LLM integration**
   ```python
   # Bad: raglet is retrieval-only
   def answer(self, query: str):
       return llm.generate(query)
   ```

4. **Skip interfaces**
   ```python
   # Bad: Direct dependency on concrete class
   def __init__(self, extractor: TextExtractor):
       pass
   ```

5. **Skip tests**
   ```python
   # Bad: No tests
   class NewFeature:
       def do_something(self):
           pass
   ```

### ✅ Do This Instead

1. **Follow SOLID**
   ```python
   # Good: Depends on interface
   def __init__(self, extractor: DocumentExtractor):
       self.extractor = extractor
   ```

2. **Keep it local**
   ```python
   # Good: Local embeddings
   def generate_embeddings(self):
       return sentence_transformers.encode(...)
   ```

3. **Retrieval only**
   ```python
   # Good: Returns chunks, user decides what to do
   def search(self, query: str) -> List[Chunk]:
       return self.vector_store.search(...)
   ```

4. **Use interfaces**
   ```python
   # Good: Depends on interface
   def __init__(self, extractor: DocumentExtractor):
       pass
   ```

5. **Write tests**
   ```python
   # Good: Has tests
   @pytest.mark.unit
   def test_new_feature():
       feature = NewFeature()
       assert feature.do_something() == expected
   ```

---

## 17. Summary

### Core Rules

1. **SOLID principles** - Always
2. **Shallow interface, deep configuration** - API simple, config deep
3. **Portable files** - `.raglet` format is core
4. **Zero infrastructure** - Everything local
5. **Retrieval only** - No LLM integration
6. **Test everything** - Unit, integration, E2E
7. **Document decisions** - Use ADR format
8. **Check scope** - Before adding features

### Key Files to Read

1. `docs/problems/00-problem-statement.md` - Why we exist
2. `plans/FINAL_PLAN.md` - What we're building
3. `docs/decisions/` - How we decided things
4. `docs/proposals/ARCHITECTURE.md` - How it's structured

### When in Doubt

1. Check `plans/FINAL_PLAN.md` for scope
2. Check `docs/decisions/` for decisions
3. Follow SOLID principles
4. Write tests
5. Ask questions (document them as proposals)

---

**Remember: raglet is portable memory. Keep it simple, keep it local, keep it focused.**
