# Configuration Exploration for raglet

## What Needs Configuration?

### Core Parameters
1. **Chunking**
   - Chunk size (tokens/chars)
   - Overlap size
   - Chunking strategy (fixed, sentence-aware, semantic)
   - Boundary preferences

2. **Embeddings**
   - Model selection (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, etc.)
   - Batch size
   - Device (CPU/GPU)

3. **Search**
   - Default top_k
   - Similarity threshold
   - Search strategy (exact, approximate)

4. **File Processing**
   - Supported file types
   - File size limits
   - Text extraction options

5. **Metadata**
   - Custom metadata to attach
   - Source tracking options

## Configuration Pattern Options

### Option 1: Constructor Parameters (Simple)

**Pros:**
- Simple, Pythonic
- No extra files
- Easy to understand

**Cons:**
- Verbose for many parameters
- Hard to reuse/share configs
- No validation

**Example:**
```python
rag = RAGlet.from_files(
    files=["doc1.txt", "doc2.md"],
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    top_k=5
)
```

### Option 2: Config Object (Structured)

**Pros:**
- Type-safe
- Reusable configs
- Clear separation
- Easy to validate

**Cons:**
- More boilerplate
- Extra class to learn

**Example:**
```python
from raglet import RAGlet, RAGletConfig

config = RAGletConfig(
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    default_top_k=5
)

rag = RAGlet.from_files(files=["doc1.txt"], config=config)
```

### Option 3: Config File (Persistent)

**Pros:**
- Shareable configs
- Version control friendly
- Separate from code
- Can have presets

**Cons:**
- Extra file to manage
- File format choice (YAML/JSON/TOML)
- Path resolution issues

**Example:**
```yaml
# raglet.yaml
chunking:
  size: 512
  overlap: 50
  strategy: sentence-aware

embeddings:
  model: all-MiniLM-L6-v2
  batch_size: 32

search:
  default_top_k: 5
  similarity_threshold: 0.7
```

```python
rag = RAGlet.from_files(files=["doc1.txt"], config_file="raglet.yaml")
```

### Option 4: Builder Pattern (Fluent)

**Pros:**
- Very readable
- Flexible chaining
- Good defaults

**Cons:**
- More complex implementation
- Can be verbose

**Example:**
```python
rag = (RAGlet.builder()
    .with_files(["doc1.txt", "doc2.md"])
    .chunk_size(512)
    .chunk_overlap(50)
    .embedding_model("all-MiniLM-L6-v2")
    .build())
```

### Option 5: Environment Variables (Deployment)

**Pros:**
- Good for deployment
- No code changes
- Standard pattern

**Cons:**
- Not great for development
- Hard to version control
- String parsing

**Example:**
```bash
export RAGLET_CHUNK_SIZE=512
export RAGLET_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

```python
rag = RAGlet.from_files(files=["doc1.txt"])  # Reads from env
```

### Option 6: Hybrid Approach (Recommended)

**Combine multiple patterns:**
- Constructor params for simple cases
- Config object for advanced/reusable cases
- Config file for project-level defaults
- Environment variables for deployment overrides

**Example:**
```python
# Simple case - just use defaults
rag = RAGlet.from_files(["doc1.txt"])

# Advanced case - use config object
config = RAGletConfig(chunk_size=1024, embedding_model="all-mpnet-base-v2")
rag = RAGlet.from_files(["doc1.txt"], config=config)

# Project-level - use config file
rag = RAGlet.from_files(["doc1.txt"], config_file=".raglet.yaml")

# Override with constructor
rag = RAGlet.from_files(
    ["doc1.txt"],
    config_file=".raglet.yaml",
    chunk_size=256  # Override file config
)
```

## Configuration Hierarchy (Precedence)

1. **Constructor parameters** (highest priority)
2. **Config object passed to constructor**
3. **Config file** (project-level defaults)
4. **Environment variables** (deployment defaults)
5. **Library defaults** (lowest priority)

## Configuration Storage in .raglet Files

### Should config be stored in .raglet?

**Yes** - Store config used to create the file:
- Enables reproducibility
- Allows inspection
- Helps with compatibility

**What to store:**
- Chunking parameters
- Embedding model name
- Creation timestamp
- Version info

**What NOT to store:**
- File paths (may not exist elsewhere)
- Device preferences (user-specific)
- Temporary settings

**Example metadata.json:**
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
  "chunk_count": 150
}
```

## Use Case Scenarios

### Scenario 1: Quick Script
**Need:** Fast, one-off usage
**Best:** Constructor parameters or defaults
```python
rag = RAGlet.from_files(["notes.txt"])
```

### Scenario 2: Project with Team
**Need:** Consistent config across team
**Best:** Config file in repo
```python
rag = RAGlet.from_files(["docs/"], config_file=".raglet.yaml")
```

### Scenario 3: Agent/Tool Integration
**Need:** Programmatic, reusable configs
**Best:** Config object
```python
codebase_config = RAGletConfig(
    chunk_size=1024,
    embedding_model="all-mpnet-base-v2"
)
rag = RAGlet.from_files(files, config=codebase_config)
```

### Scenario 4: Production Deployment
**Need:** Environment-specific overrides
**Best:** Environment variables + config file
```bash
export RAGLET_EMBEDDING_MODEL=all-mpnet-base-v2
```
```python
rag = RAGlet.from_files(files, config_file="config.yaml")
```

### Scenario 5: Experimentation
**Need:** Try different settings
**Best:** Config object with easy changes
```python
for chunk_size in [256, 512, 1024]:
    config = RAGletConfig(chunk_size=chunk_size)
    rag = RAGlet.from_files(files, config=config)
    # Test retrieval quality
```

## Recommended Design

### Core API (Simple)
```python
# Defaults - opinionated, just works
rag = RAGlet.from_files(["doc1.txt"])

# Override specific params
rag = RAGlet.from_files(
    ["doc1.txt"],
    chunk_size=1024,
    top_k=10
)
```

### Advanced API (Structured)
```python
from raglet import RAGlet, RAGletConfig

# Create reusable config
config = RAGletConfig(
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    default_top_k=5
)

# Use config
rag = RAGlet.from_files(["doc1.txt"], config=config)

# Save config for reuse
config.save("my_config.yaml")

# Load config
config = RAGletConfig.load("my_config.yaml")
```

### Config File Support
```python
# Use project config file
rag = RAGlet.from_files(["doc1.txt"], config_file=".raglet.yaml")

# Config file format (YAML)
chunking:
  size: 512
  overlap: 50

embeddings:
  model: all-MiniLM-L6-v2

search:
  default_top_k: 5
```

### Environment Variables (Optional)
```bash
# Set defaults via env vars
export RAGLET_CHUNK_SIZE=512
export RAGLET_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Implementation Structure

```
raglet/
├── config.py              # RAGletConfig class
├── defaults.py           # Default values
├── validators.py         # Config validation
└── presets.py            # Predefined configs
```

### Config Class Design

```python
@dataclass
class RAGletConfig:
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunk_strategy: str = "sentence-aware"
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"
    
    # Search
    default_top_k: int = 5
    similarity_threshold: Optional[float] = None
    
    # File processing
    max_file_size_mb: int = 10
    supported_extensions: List[str] = None
    
    # Metadata
    custom_metadata: Dict[str, Any] = None
    
    @classmethod
    def from_file(cls, path: str) -> "RAGletConfig":
        """Load config from YAML/JSON file."""
        
    def save(self, path: str) -> None:
        """Save config to file."""
        
    def validate(self) -> None:
        """Validate config values."""
        
    def merge(self, other: "RAGletConfig") -> "RAGletConfig":
        """Merge with another config (for overrides)."""
```

## Questions to Answer

1. **Should config be required or optional?**
   - Recommendation: Optional, with good defaults

2. **Should config be stored in .raglet files?**
   - Recommendation: Yes, for reproducibility

3. **What file format for config files?**
   - Recommendation: YAML (human-readable, common)

4. **Should there be presets?**
   - Recommendation: Yes (e.g., "code", "documents", "conversations")

5. **How to handle config validation?**
   - Recommendation: Validate on creation, clear error messages

6. **Should search config be separate from creation config?**
   - Recommendation: Yes, search can have different defaults

## Preset Configurations

```python
# Codebase preset
codebase_config = RAGletConfig.preset("codebase")
# chunk_size=1024, sentence-aware, all-mpnet-base-v2

# Documents preset
docs_config = RAGletConfig.preset("documents")
# chunk_size=512, all-MiniLM-L6-v2

# Conversations preset
chat_config = RAGletConfig.preset("conversations")
# chunk_size=256, preserve message boundaries
```

## Next Steps

1. Decide on primary pattern (recommend: Hybrid)
2. Design RAGletConfig class
3. Implement config file loading (YAML)
4. Add preset configurations
5. Document configuration options
6. Add validation and error handling
