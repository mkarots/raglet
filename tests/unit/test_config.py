"""Unit tests for configuration."""

import pytest

from raglet.config.config import (
    ChunkingConfig,
    EmbeddingConfig,
    RAGletConfig,
    SearchConfig,
)


class TestChunkingConfig:
    """Test ChunkingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()

        assert config.size == 256
        assert config.overlap == 50
        assert config.strategy == "sentence-aware"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(size=1024, overlap=100, strategy="fixed")

        assert config.size == 1024
        assert config.overlap == 100
        assert config.strategy == "fixed"

    def test_validate_valid_config(self):
        """Test validation of valid config."""
        config = ChunkingConfig(size=256, overlap=50)
        config.validate()  # Should not raise

    def test_validate_invalid_size(self):
        """Test validation fails for invalid size."""
        config = ChunkingConfig(size=0)

        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            config.validate()

    def test_validate_negative_overlap(self):
        """Test validation fails for negative overlap."""
        config = ChunkingConfig(overlap=-1)

        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            config.validate()

    def test_validate_overlap_too_large(self):
        """Test validation fails when overlap >= size."""
        config = ChunkingConfig(size=100, overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            config.validate()

    def test_validate_invalid_strategy(self):
        """Test validation fails for invalid strategy."""
        config = ChunkingConfig(strategy="invalid")

        with pytest.raises(ValueError, match="Invalid chunk_strategy"):
            config.validate()


class TestRAGletConfig:
    """Test RAGletConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RAGletConfig()

        assert isinstance(config.chunking, ChunkingConfig)
        assert config.chunking.size == 256
        assert config.custom_metadata == {}

    def test_custom_chunking_config(self):
        """Test custom chunking configuration."""
        chunking_config = ChunkingConfig(size=1024)
        config = RAGletConfig(chunking=chunking_config)

        assert config.chunking.size == 1024

    def test_custom_metadata(self):
        """Test custom metadata."""
        metadata = {"project": "test", "version": "1.0"}
        config = RAGletConfig(custom_metadata=metadata)

        assert config.custom_metadata == metadata

    def test_validate(self):
        """Test config validation."""
        config = RAGletConfig()
        config.validate()  # Should not raise

        # Test that invalid chunking config is caught
        config.chunking.size = 0
        with pytest.raises(ValueError):
            config.validate()


class TestConfigSerialization:
    """Test config serialization (to_dict/from_dict)."""

    def test_chunking_config_to_dict(self):
        """Test ChunkingConfig.to_dict()."""
        config = ChunkingConfig(size=1024, overlap=100, strategy="fixed")
        data = config.to_dict()

        assert data == {
            "size": 1024,
            "overlap": 100,
            "strategy": "fixed",
        }

    def test_chunking_config_from_dict(self):
        """Test ChunkingConfig.from_dict()."""
        data = {"size": 1024, "overlap": 100, "strategy": "fixed"}
        config = ChunkingConfig.from_dict(data)

        assert config.size == 1024
        assert config.overlap == 100
        assert config.strategy == "fixed"

    def test_chunking_config_roundtrip(self):
        """Test ChunkingConfig serialization roundtrip."""
        original = ChunkingConfig(size=1024, overlap=100, strategy="fixed")
        data = original.to_dict()
        restored = ChunkingConfig.from_dict(data)

        assert restored.size == original.size
        assert restored.overlap == original.overlap
        assert restored.strategy == original.strategy

    def test_chunking_config_from_dict_defaults(self):
        """Test ChunkingConfig.from_dict() uses defaults."""
        config = ChunkingConfig.from_dict({})

        assert config.size == 256
        assert config.overlap == 50
        assert config.strategy == "sentence-aware"

    def test_embedding_config_to_dict(self):
        """Test EmbeddingConfig.to_dict()."""
        config = EmbeddingConfig(
            model="test-model",
            batch_size=64,
            device="cuda",
            normalize=True,
            use_fp16=False,
        )
        data = config.to_dict()

        assert data == {
            "model": "test-model",
            "batch_size": 64,
            "device": "cuda",
            "normalize": True,
            "use_fp16": False,
            "torch_compile": False,
        }

    def test_embedding_config_from_dict(self):
        """Test EmbeddingConfig.from_dict()."""
        data = {
            "model": "test-model",
            "batch_size": 64,
            "device": "cuda",
            "normalize": True,
        }
        config = EmbeddingConfig.from_dict(data)

        assert config.model == "test-model"
        assert config.batch_size == 64
        assert config.device == "cuda"
        assert config.normalize is True

    def test_embedding_config_roundtrip(self):
        """Test EmbeddingConfig serialization roundtrip."""
        original = EmbeddingConfig(model="test-model", batch_size=64, device="cuda", normalize=True)
        data = original.to_dict()
        restored = EmbeddingConfig.from_dict(data)

        assert restored.model == original.model
        assert restored.batch_size == original.batch_size
        assert restored.device == original.device
        assert restored.normalize == original.normalize

    def test_search_config_to_dict(self):
        """Test SearchConfig.to_dict()."""
        config = SearchConfig(default_top_k=10, similarity_threshold=0.7)
        data = config.to_dict()

        assert data == {
            "default_top_k": 10,
            "similarity_threshold": 0.7,
            "index_type": "flat_ip",
        }

    def test_search_config_to_dict_no_threshold(self):
        """Test SearchConfig.to_dict() without threshold."""
        config = SearchConfig(default_top_k=10, similarity_threshold=None)
        data = config.to_dict()

        assert "similarity_threshold" not in data
        assert data["default_top_k"] == 10

    def test_search_config_from_dict(self):
        """Test SearchConfig.from_dict()."""
        data = {"default_top_k": 10, "similarity_threshold": 0.7}
        config = SearchConfig.from_dict(data)

        assert config.default_top_k == 10
        assert config.similarity_threshold == 0.7

    def test_search_config_from_dict_no_threshold(self):
        """Test SearchConfig.from_dict() without threshold."""
        data = {"default_top_k": 10}
        config = SearchConfig.from_dict(data)

        assert config.default_top_k == 10
        assert config.similarity_threshold is None

    def test_raglet_config_to_dict(self):
        """Test RAGletConfig.to_dict() includes nested configs."""
        config = RAGletConfig()
        config.chunking.size = 1024
        config.embedding.model = "test-model"
        config.search.default_top_k = 10
        config.custom_metadata = {"project": "test"}

        data = config.to_dict()

        assert "chunking" in data
        assert "embedding" in data
        assert "search" in data
        assert "custom_metadata" in data
        assert data["chunking"]["size"] == 1024
        assert data["embedding"]["model"] == "test-model"
        assert data["search"]["default_top_k"] == 10
        assert data["custom_metadata"] == {"project": "test"}

    def test_raglet_config_from_dict(self):
        """Test RAGletConfig.from_dict() handles nested configs."""
        data = {
            "chunking": {"size": 1024, "overlap": 100},
            "embedding": {"model": "test-model"},
            "search": {"default_top_k": 10},
            "custom_metadata": {"project": "test"},
        }
        config = RAGletConfig.from_dict(data)

        assert config.chunking.size == 1024
        assert config.chunking.overlap == 100
        assert config.embedding.model == "test-model"
        assert config.search.default_top_k == 10
        assert config.custom_metadata == {"project": "test"}

    def test_raglet_config_roundtrip(self):
        """Test RAGletConfig serialization roundtrip."""
        original = RAGletConfig()
        original.chunking.size = 1024
        original.embedding.model = "test-model"
        original.search.default_top_k = 10
        original.custom_metadata = {"project": "test"}

        data = original.to_dict()
        restored = RAGletConfig.from_dict(data)

        assert restored.chunking.size == original.chunking.size
        assert restored.embedding.model == original.embedding.model
        assert restored.search.default_top_k == original.search.default_top_k
        assert restored.custom_metadata == original.custom_metadata

    def test_raglet_config_from_dict_defaults(self):
        """Test RAGletConfig.from_dict() uses defaults for missing keys."""
        config = RAGletConfig.from_dict({})

        assert config.chunking.size == 256
        assert config.embedding.model == "all-MiniLM-L6-v2"
        assert config.search.default_top_k == 5
        assert config.custom_metadata == {}
