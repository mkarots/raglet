# Deep Configuration Proposal for raglet

## Philosophy: "Shallow Interfaces, Deep Configuration"

**Core Principle:**
- **Shallow Interface** - Simple API that "just works" for 80% of users
- **Deep Configuration** - When you need to customize, configuration is your escape hatch - ergonomic, sufficient, and powerful

**The API stays simple, but configuration goes deep.**

## Goal

Create a configuration system that:
- **Ergonomic** - Natural, intuitive way to configure everything
- **Sufficient** - Can configure every aspect of behavior
- **Deep** - No artificial limits, configure as much as needed
- **Portable** - Configs travel with project (files, not code)
- **Zero infrastructure** - No config servers, just files and objects

---

## Current State

**Basic Configuration:**
- Flat dataclass with simple parameters
- Constructor params, config object, config file
- Basic validation

**Limitations:**
- No nested/hierarchical configs
- No per-component configuration
- Limited validation
- No configuration inheritance
- No per-file-type settings

---

## Proposed: Deep Configuration System

### Design Principles

1. **Layered Configuration** - Nested configs for different components
2. **Schema Validation** - Strong typing and validation (Pydantic-based)
3. **Inheritance** - Base configs that can be extended
4. **Per-Component** - Separate configs for chunking, embeddings, search
5. **Per-File-Type** - Different settings for different file types
6. **Extensible** - Plugin system for custom configurations

---

## Architecture

### Nested Configuration Structure

```python
@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    size: int = 512
    overlap: int = 50
    strategy: str = "sentence-aware"  # "fixed", "sentence-aware", "semantic"
    preserve_paragraphs: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 2048
    chunk_separators: List[str] = None  # Custom separators
    
    def validate(self) -> None:
        """Validate chunking configuration."""
        if self.size < self.min_chunk_size:
            raise ValueError(f"chunk_size must be >= {self.min_chunk_size}")
        if self.overlap >= self.size:
            raise ValueError("chunk_overlap must be < chunk_size")

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"  # "cpu", "cuda", "mps"
    normalize_embeddings: bool = True
    show_progress: bool = False
    cache_dir: Optional[str] = None
    
    def validate(self) -> None:
        """Validate embedding configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

@dataclass
class SearchConfig:
    """Configuration for search/retrieval."""
    default_top_k: int = 5
    similarity_threshold: Optional[float] = None
    rerank: bool = False
    rerank_model: Optional[str] = None
    return_metadata: bool = True
    return_scores: bool = True
    
    def validate(self) -> None:
        """Validate search configuration."""
        if self.default_top_k < 1:
            raise ValueError("default_top_k must be >= 1")

@dataclass
class FileProcessingConfig:
    """Configuration for file processing."""
    max_file_size_mb: int = 10
    supported_extensions: List[str] = None
    text_extraction: Dict[str, Any] = None  # Per-file-type settings
    encoding: str = "utf-8"
    error_handling: str = "skip"  # "skip", "raise", "warn"
    
    def validate(self) -> None:
        """Validate file processing configuration."""

@dataclass
class RAGletConfig:
    """Main configuration class with nested components."""
    # Nested configs
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    file_processing: FileProcessingConfig = field(default_factory=FileProcessingConfig)
    
    # Global settings
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
    # Configuration metadata
    name: Optional[str] = None
    description: Optional[str] = None
    extends: Optional[str] = None  # Config inheritance
    
    def validate(self) -> None:
        """Validate entire configuration."""
        self.chunking.validate()
        self.embeddings.validate()
        self.search.validate()
        self.file_processing.validate()
    
    def merge(self, other: "RAGletConfig") -> "RAGletConfig":
        """Deep merge with another config."""
        # Merge nested configs recursively
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGletConfig":
        """Create config from dictionary (nested)."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        ...
```

---

## Usage Examples: Shallow Interface, Deep Configuration

### Shallow Interface - Simple API

```python
# Level 1: Simplest - just works
rag = RAGlet.from_files(["doc.txt"])

# Level 2: Override a few common params - still simple
rag = RAGlet.from_files(["doc.txt"], chunk_size=1024, top_k=10)

# Level 3: Use preset - simple, but gets you closer to your use case
rag = RAGlet.from_files(["codebase/"], preset="codebase")

# That's it - API doesn't get more complex than this
```

**The API surface is intentionally shallow.** No nested parameters, no complex options.

### Deep Configuration - Your Escape Hatch

When you need more control, configuration is where you go:

```python
# Level 4: Config object - start customizing
config = RAGletConfig(chunk_size=1024)
rag = RAGlet.from_files(["doc.txt"], config=config)

# Level 5: Deep config - customize components
config = RAGletConfig(
    chunking=ChunkingConfig(size=1024, strategy="semantic"),
    embeddings=EmbeddingConfig(model="all-mpnet-base-v2")
)
rag = RAGlet.from_files(["doc.txt"], config=config)

# Level 6: Very deep config - customize everything
config = RAGletConfig(
    chunking=ChunkingConfig(
        size=1024,
        overlap=100,
        strategy="semantic",
        preserve_paragraphs=True,
        min_chunk_size=100,
        max_chunk_size=2048,
        chunk_separators=["\n\n", "\n---\n"],
        respect_sentence_boundaries=True,
        respect_word_boundaries=True,
        metadata_extraction=True
    ),
    embeddings=EmbeddingConfig(
        model="all-mpnet-base-v2",
        batch_size=64,
        device="cuda",
        normalize_embeddings=True,
        show_progress=True,
        cache_dir="~/.raglet/cache",
        model_kwargs={"trust_remote_code": True}
    ),
    search=SearchConfig(
        default_top_k=10,
        similarity_threshold=0.7,
        rerank=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        return_metadata=True,
        return_scores=True,
        score_threshold=0.5,
        max_results=100
    ),
    file_processing=FileProcessingConfig(
        max_file_size_mb=10,
        supported_extensions=[".txt", ".md", ".py"],
        encoding="utf-8",
        error_handling="warn",
        text_extraction={
            "pdf": {
                "extract_images": False,
                "extract_tables": True,
                "extract_metadata": True,
                "page_range": None
            },
            "html": {
                "remove_scripts": True,
                "remove_styles": True,
                "preserve_links": False,
                "base_url": None
            },
            "docx": {
                "extract_comments": False,
                "extract_headers": True,
                "extract_footers": False
            }
        }
    ),
    custom_metadata={
        "project": "myproject",
        "environment": "production",
        "version": "1.0.0"
    }
)

rag = RAGlet.from_files(["doc.txt"], config=config)
```

**Configuration goes as deep as you need.** No limits, fully ergonomic.

### Ergonomic Configuration Patterns

**Pattern 1: Override Specific Component**
```python
# Override just chunking - other components use defaults
config = RAGletConfig(
    chunking=ChunkingConfig(size=1024, overlap=100)
)
rag = RAGlet.from_files(["doc.txt"], config=config)
```

**Pattern 2: Start from Preset, Then Customize**
```python
# Start with preset, then go deeper
config = RAGletConfig.preset("codebase")
config.chunking.size = 2048  # Override preset
config.embeddings.device = "cuda"  # Override preset
rag = RAGlet.from_files(["codebase/"], config=config)
```

**Pattern 3: Load Base Config, Override Parts**
```python
# Load team config, override for your use case
base_config = RAGletConfig.load("team-config.yaml")
my_config = RAGletConfig(
    extends=base_config,  # Inherit base
    chunking=ChunkingConfig(size=2048)  # Override chunking
)
rag = RAGlet.from_files(["docs/"], config=my_config)
```

**Pattern 4: Compose Config Components**
```python
# Reuse config components
chunking_for_code = ChunkingConfig(size=1024, strategy="sentence-aware")
embeddings_fast = EmbeddingConfig(model="all-MiniLM-L6-v2", device="cpu")

# Mix and match
config1 = RAGletConfig(chunking=chunking_for_code, embeddings=embeddings_fast)
config2 = RAGletConfig(chunking=chunking_for_code)  # Reuse chunking config
```

### Deep Configuration via Dict
```python
# Deep configuration via YAML/dict
config_dict = {
    "chunking": {
        "size": 1024,
        "overlap": 100,
        "strategy": "semantic",
        "preserve_paragraphs": True
    },
    "embeddings": {
        "model": "all-mpnet-base-v2",
        "batch_size": 64,
        "device": "cuda"
    },
    "search": {
        "default_top_k": 10,
        "similarity_threshold": 0.7,
        "rerank": True
    }
}
config = RAGletConfig.from_dict(config_dict)
rag = RAGlet.from_files(["doc.txt"], config=config)
```

### Configuration File (YAML)
```yaml
# .raglet.yaml
name: "codebase-config"
description: "Configuration for codebase indexing"

chunking:
  size: 1024
  overlap: 100
  strategy: "sentence-aware"
  preserve_paragraphs: true
  min_chunk_size: 100

embeddings:
  model: "all-mpnet-base-v2"
  batch_size: 64
  device: "cpu"
  normalize_embeddings: true

search:
  default_top_k: 10
  similarity_threshold: 0.7
  return_metadata: true

file_processing:
  max_file_size_mb: 10
  supported_extensions: [".py", ".md", ".txt"]
  encoding: "utf-8"
  error_handling: "warn"

custom_metadata:
  project: "myproject"
  version: "1.0.0"
```

### Configuration Inheritance
```python
# Base config
base_config = RAGletConfig.load("base.yaml")

# Extend base config
extended_config = RAGletConfig(
    extends="base.yaml",  # Inherit from base
    chunking=ChunkingConfig(size=2048)  # Override chunking
)
```

### Per-File-Type Configuration
```python
config = RAGletConfig(
    file_processing=FileProcessingConfig(
        text_extraction={
            "pdf": {
                "extract_images": False,
                "extract_tables": True
            },
            "html": {
                "remove_scripts": True,
                "remove_styles": True
            },
            "docx": {
                "extract_comments": False
            }
        }
    )
)
```

---

## Advanced Features

### 1. Schema Validation (Pydantic-based)

```python
from pydantic import BaseModel, Field, validator

class ChunkingConfig(BaseModel):
    size: int = Field(default=512, ge=50, le=2048)
    overlap: int = Field(default=50, ge=0)
    strategy: str = Field(default="sentence-aware", regex="^(fixed|sentence-aware|semantic)$")
    
    @validator('overlap')
    def validate_overlap(cls, v, values):
        if 'size' in values and v >= values['size']:
            raise ValueError('overlap must be < size')
        return v
```

**Benefits:**
- Automatic validation
- Type coercion
- Clear error messages
- IDE support

### 2. Configuration Presets with Deep Customization

```python
# Preset with overrides
config = RAGletConfig.preset("codebase")
config.chunking.size = 2048  # Override preset
config.embeddings.model = "all-mpnet-base-v2"  # Override preset
```

### 3. Configuration Templates

```python
# Template system
templates = {
    "codebase": {
        "chunking": {"size": 1024, "strategy": "sentence-aware"},
        "embeddings": {"model": "all-mpnet-base-v2"}
    },
    "documents": {
        "chunking": {"size": 512, "strategy": "fixed"},
        "embeddings": {"model": "all-MiniLM-L6-v2"}
    }
}

config = RAGletConfig.from_template("codebase")
```

### 4. Environment-Specific Configs

```python
# Load environment-specific config
config = RAGletConfig.load("config.dev.yaml")  # Development
config = RAGletConfig.load("config.prod.yaml")  # Production

# Or via environment variable
import os
env = os.getenv("RAGLET_ENV", "dev")
config = RAGletConfig.load(f"config.{env}.yaml")
```

### 5. Configuration Diff/Merge

```python
# Compare configs
diff = config1.diff(config2)
# Returns: {"chunking.size": (512, 1024), ...}

# Merge configs intelligently
merged = config1.merge(config2, strategy="deep")  # Deep merge nested dicts
```

---

## Implementation Plan

### Phase 1: Nested Config Structure
- [ ] Create nested config classes (ChunkingConfig, EmbeddingConfig, etc.)
- [ ] Update RAGletConfig to use nested configs
- [ ] Maintain backward compatibility (flat config still works)
- [ ] Add validation to each config class

### Phase 2: Schema Validation
- [ ] Integrate Pydantic (or use dataclass validation)
- [ ] Add field validators
- [ ] Add custom validators
- [ ] Improve error messages

### Phase 3: Configuration Loading
- [ ] Support nested YAML/JSON loading
- [ ] Add from_dict() method with nested support
- [ ] Add to_dict() method
- [ ] Support configuration inheritance (extends)

### Phase 4: Advanced Features
- [ ] Per-file-type configuration
- [ ] Configuration templates
- [ ] Configuration diff/merge
- [ ] Environment-specific configs

### Phase 5: Documentation
- [ ] Document all configuration options
- [ ] Add examples for deep configuration
- [ ] Create configuration reference guide
- [ ] Add migration guide from flat to nested

---

## Backward Compatibility

### Migration Path

**Old (Still Works):**
```python
# Flat config
config = RAGletConfig(chunk_size=512, embedding_model="all-MiniLM-L6-v2")
```

**New (Nested):**
```python
# Nested config
config = RAGletConfig(
    chunking=ChunkingConfig(size=512),
    embeddings=EmbeddingConfig(model="all-MiniLM-L6-v2")
)
```

**Auto-Conversion:**
```python
# Automatically convert flat to nested
config = RAGletConfig(chunk_size=512)  # Flat
# Internally converts to: config.chunking.size = 512
```

---

## Configuration File Format

### YAML Example (Nested)

```yaml
# .raglet.yaml
name: "my-config"
version: "1.0.0"

chunking:
  size: 512
  overlap: 50
  strategy: "sentence-aware"
  preserve_paragraphs: true
  min_chunk_size: 50
  max_chunk_size: 2048

embeddings:
  model: "all-MiniLM-L6-v2"
  batch_size: 32
  device: "cpu"
  normalize_embeddings: true

search:
  default_top_k: 5
  similarity_threshold: null
  rerank: false
  return_metadata: true

file_processing:
  max_file_size_mb: 10
  supported_extensions: [".txt", ".md", ".pdf", ".html", ".docx"]
  encoding: "utf-8"
  error_handling: "skip"
  text_extraction:
    pdf:
      extract_images: false
      extract_tables: true
    html:
      remove_scripts: true
      remove_styles: true

custom_metadata:
  project: "myproject"
  environment: "development"
```

---

## Benefits: Why Deep Configuration Matters

### For Simple Use Cases
- **Shallow interface** - API stays simple, just works
- **Defaults are excellent** - Opinionated choices work well
- **No configuration needed** - 80% of users never touch config

### For Advanced Use Cases
- **Deep customization** - Configure everything you need
- **Ergonomic** - Natural, intuitive configuration structure
- **Sufficient** - No artificial limits, configure as deep as needed
- **Portable** - Save configs, share them, version control them
- **Composable** - Mix and match config components

### For Power Users
- **Escape hatch** - Configuration is where you customize deeply
- **No API bloat** - Complex options live in config, not API
- **Reusable** - Create configs once, use everywhere
- **Extensible** - Easy to add new configuration options

## Key Insight

**The API is your happy path. Configuration is your escape hatch.**

- **Happy path:** `RAGlet.from_files(["doc.txt"])` - works for most
- **Escape hatch:** `RAGlet.from_files(["doc.txt"], config=deep_config)` - customize everything

This keeps the API clean while giving power users full control.

---

## Considerations

### Complexity vs. Simplicity

**Risk:** Deep config might be too complex for simple use cases

**Mitigation:**
- Maintain simple defaults (80% use case)
- Auto-convert flat to nested (backward compatible)
- Clear documentation with examples
- Presets for common use cases

### Validation Overhead

**Risk:** Too much validation might slow down startup

**Mitigation:**
- Lazy validation (only when needed)
- Fast validation (Pydantic is fast)
- Optional strict mode

### Learning Curve

**Risk:** Users might not understand nested configs

**Mitigation:**
- Simple examples first
- Progressive disclosure (start simple, go deep)
- Good error messages
- Migration guide

---

## Design Philosophy: Shallow Interface, Deep Configuration

### Shallow Interface (API)

```python
# Simple - just works
rag = RAGlet.from_files(["doc.txt"])

# Still simple - override what you need
rag = RAGlet.from_files(["doc.txt"], chunk_size=1024)

# That's it - API doesn't get more complex
```

**The API surface stays minimal.** No nested parameters, no complex options in the constructor.

### Deep Configuration (Escape Hatch)

```python
# When you need deep customization, use configuration
config = RAGletConfig(
    chunking=ChunkingConfig(
        size=1024,
        overlap=100,
        strategy="semantic",
        preserve_paragraphs=True,
        min_chunk_size=100,
        max_chunk_size=2048,
        chunk_separators=["\n\n", "\n---\n"],
        respect_sentence_boundaries=True,
        respect_word_boundaries=True
    ),
    embeddings=EmbeddingConfig(
        model="all-mpnet-base-v2",
        batch_size=64,
        device="cuda",
        normalize_embeddings=True,
        show_progress=True,
        cache_dir="~/.raglet/cache"
    ),
    search=SearchConfig(
        default_top_k=10,
        similarity_threshold=0.7,
        rerank=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        return_metadata=True,
        return_scores=True,
        score_threshold=0.5
    ),
    file_processing=FileProcessingConfig(
        max_file_size_mb=10,
        supported_extensions=[".txt", ".md", ".py"],
        encoding="utf-8",
        error_handling="warn",
        text_extraction={
            "pdf": {
                "extract_images": False,
                "extract_tables": True,
                "extract_metadata": True
            },
            "html": {
                "remove_scripts": True,
                "remove_styles": True,
                "preserve_links": False
            }
        }
    )
)

rag = RAGlet.from_files(["doc.txt"], config=config)
```

**Configuration is where you go deep.** Every aspect is configurable, ergonomically.

## Recommendation

**Proceed with Deep Configuration** because:

1. ✅ Maintains shallow API (simple interface)
2. ✅ Enables deep customization (configuration escape hatch)
3. ✅ Ergonomic configuration (natural, intuitive)
4. ✅ Sufficient (can configure everything)
5. ✅ Backward compatible (simple still works)
6. ✅ Aligns with "shallow interface, deep configuration" principle

**Implementation Priority:**
1. **High:** Nested config structure (Phase 1)
2. **Medium:** Schema validation (Phase 2)
3. **Medium:** Configuration loading (Phase 3)
4. **Low:** Advanced features (Phase 4)

---

## Summary: Shallow Interface, Deep Configuration

### The Philosophy

```
┌─────────────────────────────────────────────────────────┐
│                    SHALLOW INTERFACE                    │
│                                                         │
│  RAGlet.from_files(["doc.txt"])                       │
│  RAGlet.from_files(["doc.txt"], chunk_size=1024)     │
│  RAGlet.from_files(["doc.txt"], preset="codebase")    │
│                                                         │
│  Simple. Clean. Just works.                           │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Need more control?
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  DEEP CONFIGURATION                     │
│                                                         │
│  config = RAGletConfig(                               │
│      chunking=ChunkingConfig(...),                     │
│      embeddings=EmbeddingConfig(...),                  │
│      search=SearchConfig(...),                         │
│      file_processing=FileProcessingConfig(...)          │
│  )                                                      │
│                                                         │
│  Ergonomic. Sufficient. Deep.                          │
└─────────────────────────────────────────────────────────┘
```

### Key Principles

1. **API Stays Simple**
   - No nested parameters in constructor
   - No complex options in method signatures
   - Common use cases work with defaults

2. **Configuration Goes Deep**
   - Every aspect is configurable
   - Natural, ergonomic structure
   - No artificial limits

3. **Progressive Disclosure**
   - Start simple: `from_files(["doc.txt"])`
   - Override common params: `chunk_size=1024`
   - Use presets: `preset="codebase"`
   - Go deep: `config=RAGletConfig(...)`

4. **Configuration is Portable**
   - Save configs to files
   - Share configs with team
   - Version control configs
   - Load configs anywhere

### Why This Works

**For 80% of users:**
- Simple API, no configuration needed
- Excellent defaults, "just works"

**For 20% of users:**
- Deep configuration when needed
- Ergonomic, sufficient, powerful
- No API bloat

**For everyone:**
- Clear separation: API vs Configuration
- Progressive complexity
- No surprises

## Next Steps

1. Review and approve this proposal
2. Update FINAL_PLAN.md to include deep configuration philosophy
3. Implement Phase 1 (nested config structure)
4. Ensure API stays shallow (no nested params in constructor)
5. Make configuration deep (every aspect configurable)
6. Add tests for nested configs
7. Update documentation with "shallow interface, deep configuration" examples
