# raglet Decision Document

**Status:** Planning & Design Phase  
**Date:** February 2026
**Version:** 1.0

---

## Executive Summary

**raglet** is a Python library that creates portable `.raglet` files - single-file knowledge bases containing chunks, embeddings, and metadata. It solves the problem of making small text corpora (codebases, conversations, notes) searchable without infrastructure complexity.

**Current Phase:** Design complete, ready for implementation  
**Decision:** Proceed with Python library implementation following the "tiny" philosophy

---

## 1. Project Vision & Problem Statement

### The Problem (from WHY.md)

There's a class of knowledge that's **small but too big for a prompt**:
- A codebase
- A Slack conversation  
- A WhatsApp chat export
- A folder of meeting notes

These are small (a few megabytes) but don't fit in a context window. They also don't justify a vector database, server, or infrastructure setup.

**Current options are inadequate:**
1. Copy-paste into LLM → works until it doesn't fit
2. Set up RAG pipeline → embeddings, vector DB, retrieval, chunking, config (way too much)
3. Just remember → scroll, grep, hope you find it

**Gap:** No lightweight, portable way to make a small pile of text searchable and LLM-ready.

### The Solution

**raglet is portable memory.**

It takes small context and turns it into a single `.raglet` file that you can:
- Save, share, commit, or carry around
- Load anywhere
- Search instantly
- Get retrieval-ready context for any LLM or tool

**No server. No API keys. No infrastructure. Just a Python object and a file.**

### Core Principles

1. **Portable** - One file. Save it, git commit it, email it, drag it to another machine
2. **Small by design** - Built for workspace-scale problems (codebases, conversations, notes). Not the internet
3. **Retrieval only** - raglet finds the right chunks. You decide what to do with them. Bring your own LLM
4. **Open format** - The `.raglet` file is easily decodable. Embeddings are extractable. No lock-in
5. **Zero infrastructure** - `pip install raglet`. That's it

---

## 2. Current Phase: Planning & Design Complete

### Phase Status: **Design Phase Complete**

**Completed:**
- ✅ Vision and principles defined (WHY.md)
- ✅ Architecture exploration (Python library vs web service)
- ✅ Configuration pattern analysis
- ✅ Agentic tool integration design
- ✅ File format specification
- ✅ API design
- ✅ Implementation roadmap

**Next Phase:** Implementation (Week 1-5)

### What We've Decided

1. **Library, not service** - Python library, not web API
2. **Portable files** - `.raglet` format is core value proposition
3. **Retrieval-only** - No LLM integration in library
4. **Opinionated defaults** - Works for 80% of users out of the box
5. **Agent-friendly** - Exposed as Anthropic tools for AI agents

---

## 3. High-Level Questions & Answers

### Q1: Is this useful? What's the value proposition?

**Answer: Yes, for a specific use case.**

**Value Proposition:**
- **For developers:** Need retrieval component that "just works" - no infrastructure setup
- **For teams:** Shareable knowledge bases as files (git-friendly)
- **For agents:** Portable memory that can be created, searched, and shared
- **For ephemeral projects:** Context that matters now, not forever

**Not for:**
- Large-scale datasets (use proper vector DBs)
- Production RAG systems requiring high availability
- Real-time updates (files are static snapshots)

**The Bet:** Most RAG problems are small. A few files. A few thousand chunks. If retrieval is as simple as `pip install` and a `.raglet` file, it becomes a building block that shows up everywhere.

### Q2: Why not use existing solutions?

**Comparison:**

| Solution | Setup Complexity | Portability | Infrastructure |
|----------|-----------------|-------------|----------------|
| **Pinecone/Weaviate** | High | Low (cloud) | Required |
| **ChromaDB/LanceDB** | Medium | Medium | Local server |
| **LangChain + FAISS** | Medium | Low | Code complexity |
| **raglet** | Low | High (single file) | None |

**Differentiation:**
- **Portability:** Single file vs. databases/servers
- **Simplicity:** `pip install` vs. infrastructure setup
- **Opinionated:** Strong defaults vs. configuration complexity
- **Retrieval-only:** Focused scope vs. full RAG pipeline

### Q3: Who is the target user?

**Primary Users:**
1. **Developers building LLM tools** - Need retrieval component
2. **Anyone with small text corpus** - Want searchability without infrastructure
3. **AI agents** - Need portable memory/knowledge bases
4. **Ephemeral projects** - Context that matters now, not forever

**Not for:**
- ML engineers building production RAG systems
- Large-scale document processing
- Real-time document indexing

### Q4: What makes this different from LangChain/LLamaIndex?

**Key Differences:**

| Aspect | LangChain/LLamaIndex | raglet |
|--------|---------------------|---------|
| **Scope** | Full RAG pipeline | Retrieval only |
| **Portability** | Code + config | Single file |
| **Complexity** | Many components | One class |
| **Infrastructure** | Often requires DBs | Zero |
| **Use Case** | Production systems | Workspace-scale |

**raglet is complementary:** Use raglet for portable knowledge bases, LangChain for full pipelines.

### Q5: Is the file format a good idea?

**Yes, for these reasons:**

**Advantages:**
- **Portability** - Email, git commit, share via any channel
- **Reproducibility** - Same file, same results
- **Version control** - Track knowledge base changes
- **No dependencies** - File contains everything needed
- **Open format** - Decodable without library

**Trade-offs:**
- **Static** - Can't update without recreating
- **File size** - Larger than raw text (embeddings + index)
- **No real-time** - Not suitable for live updates

**Verdict:** File format is core to the value proposition. Trade-offs are acceptable for the use case.

### Q6: Will developers actually use this?

**Evidence for yes:**
- Developers already use similar patterns (SQLite files, pickle files)
- "Just works" tools are highly valued
- Portable formats are popular (Docker images, PDFs)
- Agent tools are growing rapidly

**Risks:**
- May be too simple for some use cases
- File size could be limiting
- Competition from established tools

**Mitigation:**
- Focus on specific use case (workspace-scale)
- Make defaults excellent (opinionated)
- Provide clear examples and documentation
- Integrate with agent frameworks

---

## 4. Architecture Decisions

### Decision 1: Library vs. Service

**Decision:** Python library only

**Rationale:**
- Aligns with "zero infrastructure" principle
- Portable files don't need servers
- Easier to integrate into existing workflows
- Can be used by agents directly

**Rejected:** Web service approach (from original planning docs)
- Requires infrastructure
- Doesn't match portable file vision
- Adds complexity

### Decision 2: File Format

**Decision:** Zip archive containing:
- `metadata.json` - Version, config, creation date
- `chunks.json` - Array of chunk objects
- `embeddings.npy` - NumPy array of embeddings
- `faiss_index.bin` - Serialized FAISS index
- `sources.json` - Source file metadata

**Rationale:**
- Portable (single file)
- Open format (decodable)
- Efficient (compressed)
- Complete (everything needed)

### Decision 3: Technology Stack

**Core Dependencies:**
- `sentence-transformers` - Embeddings (local, CPU-friendly)
- `faiss-cpu` - Vector search (no external DB)
- `numpy` - Array operations

**Document Processing:**
- `PyPDF2` - PDF extraction
- `beautifulsoup4` - HTML parsing
- `python-docx` - DOCX extraction

**Rationale:**
- All local, no external services
- CPU-friendly (works everywhere)
- Mature, well-maintained libraries
- Small footprint

### Decision 4: Configuration Approach

**Decision:** Hybrid with smart defaults

**Patterns:**
1. **Constructor params** (primary) - Simple, Pythonic
2. **Config object** (secondary) - Reusable configs
3. **Config file** (tertiary) - Team consistency
4. **Presets** - Common use cases

**Rationale:**
- 80% of users: just works with defaults
- 20% of users: can override what they need
- Teams: can share configs via files
- Agents: can use config objects programmatically

**Precedence:** Constructor > Config object > Config file > Env vars > Defaults

### Decision 5: Agent Integration

**Decision:** Expose as Anthropic tools

**Tools:**
1. `create_raglet` - Create .raglet file from documents
2. `search_raglet` - Search a .raglet file
3. `get_raglet_info` - Get metadata about file

**Rationale:**
- Agents need programmatic access
- Tools maintain "zero infrastructure" (library, not API)
- Portable files work well for agent workflows
- Can be adapted to other frameworks

---

## 5. API Design

### Core API

```python
from raglet import RAGlet

# Simplest - just works
rag = RAGlet.from_files(["doc.txt"])

# Override defaults
rag = RAGlet.from_files(
    ["doc.txt"],
    chunk_size=1024,
    top_k=10
)

# Use preset
rag = RAGlet.from_files(["codebase/"], preset="codebase")

# Save portable file
rag.save("knowledge.raglet")

# Load later
rag = RAGlet.load("knowledge.raglet")

# Search
results = rag.search("what is X?", top_k=5)
```

### Configuration API

```python
from raglet import RAGletConfig

# Create reusable config
config = RAGletConfig(
    chunk_size=512,
    embedding_model="all-mpnet-base-v2"
)

# Use config
rag = RAGlet.from_files(["doc.txt"], config=config)

# Save/load config
config.save("project_config.yaml")
config = RAGletConfig.load("project_config.yaml")
```

### Agent Tools API

```python
from raglet.tools import (
    create_raglet,
    search_raglet,
    get_raglet_info
)

# Agent can call these functions
result = create_raglet(
    file_paths=["doc1.txt", "doc2.md"],
    output_path="knowledge.raglet"
)

results = search_raglet(
    file_path="knowledge.raglet",
    query="what is X?",
    top_k=5
)
```

---

## 6. Implementation Roadmap

### Phase 1: Core Library (Week 1)
- Package structure
- Document processor (.txt, .md)
- Chunker (512 tokens, 50 overlap)
- Basic RAGlet class

### Phase 2: Embeddings & Search (Week 2)
- sentence-transformers integration
- FAISS wrapper
- Search/retrieval API
- Unit tests

### Phase 3: File Format (Week 3)
- .raglet serialization
- Save/load methods
- Compression
- Version handling

### Phase 4: Document Support (Week 4)
- PDF, HTML, DOCX support
- Error handling
- Integration tests

### Phase 5: Polish & Publish (Week 5)
- Configuration system
- Presets
- Agent tools
- Documentation
- PyPI publish

---

## 7. Success Criteria

### MVP Success
- ✅ Create `.raglet` file from 5 text files in <10 seconds
- ✅ Load `.raglet` file and search in <1 second
- ✅ Accurate retrieval (top-5 chunks are relevant)
- ✅ File size reasonable (<10MB for 1000 chunks)
- ✅ Works on macOS, Linux, Windows

### Library Quality
- ✅ Clean, intuitive API
- ✅ Comprehensive error handling
- ✅ Good documentation
- ✅ Unit test coverage >80%
- ✅ Published to PyPI

### Adoption Indicators
- ✅ Used in at least 3 different projects
- ✅ Positive feedback on simplicity
- ✅ Agent integrations working
- ✅ Community contributions

---

## 8. Risks & Mitigations

### Risk 1: File Size Limitations

**Risk:** Large knowledge bases create large files

**Mitigation:**
- Set expectations (workspace-scale, not internet-scale)
- Document size limits
- Provide compression options
- Consider streaming for very large files (future)

### Risk 2: Competition from Established Tools

**Risk:** LangChain, LLamaIndex already exist

**Mitigation:**
- Focus on differentiation (portability, simplicity)
- Position as complementary, not replacement
- Target specific use case (workspace-scale, portable)

### Risk 3: Adoption Challenges

**Risk:** Developers may not see value

**Mitigation:**
- Clear examples and documentation
- Agent integrations (growing market)
- Focus on "just works" experience
- Community engagement

### Risk 4: Technical Complexity

**Risk:** Implementation may be harder than expected

**Mitigation:**
- Start simple (MVP first)
- Use proven libraries (sentence-transformers, FAISS)
- Iterate based on feedback
- Keep scope focused

---

## 9. Open Questions

### Q1: Should we support streaming/chunked loading?

**Status:** Defer to future  
**Rationale:** MVP focuses on small files. Can add later if needed.

### Q2: Should we support multiple embedding models per file?

**Status:** No for MVP  
**Rationale:** Adds complexity. One model per file is simpler.

### Q3: Should we provide CLI tools?

**Status:** Future enhancement  
**Rationale:** Library first. CLI can be added later.

### Q4: Should we support incremental updates?

**Status:** No for MVP  
**Rationale:** Files are static snapshots. Recreate for updates.

### Q5: Should we support custom chunkers?

**Status:** Maybe (advanced feature)  
**Rationale:** Default chunker should work for most. Can add plugin system later.

---

## 10. Next Steps

### Immediate (Week 1)
1. Set up Python package structure
2. Implement document processor
3. Implement chunker
4. Create basic RAGlet class

### Short-term (Weeks 2-3)
1. Integrate embeddings
2. Implement FAISS wrapper
3. Design file format
4. Implement save/load

### Medium-term (Weeks 4-5)
1. Add document format support
2. Implement configuration system
3. Add agent tools
4. Write documentation
5. Publish to PyPI

### Long-term (Future)
1. CLI tools
2. Streaming support
3. Custom chunkers
4. Additional embedding models
5. Community contributions

---

## 11. Decision: Proceed with Implementation

### Recommendation: **Proceed**

**Rationale:**
- Clear problem statement and solution
- Well-defined architecture
- Focused scope (retrieval-only, portable files)
- Strong differentiation (portability, simplicity)
- Growing market (agents, LLM tools)

**Confidence Level:** High

**Key Success Factors:**
1. Excellent defaults (opinionated)
2. Clear documentation
3. Agent integrations
4. Community engagement

**Go/No-Go Criteria:**
- ✅ Problem clearly defined
- ✅ Solution well-designed
- ✅ Architecture feasible
- ✅ Scope appropriate
- ✅ Differentiation clear

**Decision:** **GO** - Proceed with implementation following the roadmap.

---

## Appendix: Related Documents

- **WHY.md** - Vision and principles
- **CONFIG_EXPLORATION.md** - Configuration pattern analysis
- **CONFIG_RECOMMENDATION.md** - Configuration decisions
- **CONFIG_EXAMPLES.md** - Configuration usage examples
- **raglet_python_library_roadmap.md** - Detailed implementation plan

---

**Document Status:** Approved for implementation  
**Next Review:** After MVP completion (Week 5)
