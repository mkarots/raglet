"""Gate: configuration system works as documented in the README."""

import os
import tempfile

SECTION = "Configuration"


def register(gate):
    gate(SECTION, "custom-chunking",
         "RAGletConfig overrides chunk size and overlap", _custom_chunking)
    gate(SECTION, "threshold",
         "search with high similarity_threshold returns fewer results", _similarity_threshold)
    gate(SECTION, "defaults",
         "config defaults are valid and self-consistent", _config_defaults)


def _custom_chunking():
    from raglet import RAGlet, RAGletConfig

    config = RAGletConfig()
    config.chunking.size = 1024
    config.chunking.overlap = 100

    with tempfile.TemporaryDirectory() as tmp:
        sample = os.path.join(tmp, "doc.txt")
        with open(sample, "w") as f:
            f.write("Some text. " * 200)
        rag = RAGlet.from_files([sample], config=config)
        n = len(rag.get_all_chunks())
        assert n > 0
        return f"chunk_size=1024, overlap=100 → {n} chunks"


def _similarity_threshold():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        sample = os.path.join(tmp, "doc.txt")
        with open(sample, "w") as f:
            f.write("Python is a programming language used for web development and data science.")

        rag = RAGlet.from_files([sample])

        high_threshold = rag.search("python programming", top_k=10, similarity_threshold=0.99)
        low_threshold = rag.search("python programming", top_k=10, similarity_threshold=0.01)
        assert len(low_threshold) >= len(high_threshold), (
            "high threshold should return equal or fewer results"
        )
        return f"threshold 0.99→{len(high_threshold)} results, 0.01→{len(low_threshold)} results"


def _config_defaults():
    from raglet import RAGletConfig, ChunkingConfig, EmbeddingConfig, SearchConfig

    config = RAGletConfig()
    assert isinstance(config.chunking, ChunkingConfig)
    assert isinstance(config.embedding, EmbeddingConfig)
    assert isinstance(config.search, SearchConfig)
    assert config.chunking.size > 0, "default chunk size must be positive"
    assert config.chunking.overlap >= 0, "default overlap must be non-negative"
    assert config.embedding.model, "default model must be set"
    return f"size={config.chunking.size}, overlap={config.chunking.overlap}, model={config.embedding.model}"
