"""Unit tests for Milestone 2 configuration classes."""

import pytest

from raglet.config.config import (
    EmbeddingConfig,
    RAGletConfig,
    SearchConfig,
)


@pytest.mark.unit
class TestEmbeddingConfig:
    """Test EmbeddingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from raglet.config.config import _default_fp16

        config = EmbeddingConfig()
        assert config.model == "all-MiniLM-L6-v2"
        # batch_size is device-aware: 32 (CPU), 128 (MPS), 256 (CUDA)
        assert config.batch_size in [32, 128, 256]
        assert config.device in ["cpu", "cuda", "mps"]
        assert config.normalize is True
        assert config.use_fp16 is _default_fp16()
        assert config.torch_compile is False

    def test_validation_success(self):
        """Test successful validation."""
        config = EmbeddingConfig(
            model="all-MiniLM-L6-v2", batch_size=64, device="cpu", use_fp16=False,
        )
        config.validate()  # Should not raise

    def test_validation_empty_model(self):
        """Test validation fails for empty model."""
        config = EmbeddingConfig(model="")
        with pytest.raises(ValueError, match="model must be specified"):
            config.validate()

    def test_validation_invalid_batch_size(self):
        """Test validation fails for invalid batch size."""
        config = EmbeddingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            config.validate()

    def test_validation_invalid_device(self):
        """Test validation fails for invalid device."""
        config = EmbeddingConfig(device="invalid")
        with pytest.raises(ValueError, match="device must be"):
            config.validate()

    def test_validation_mps_device(self):
        """Test validation accepts 'mps' device."""
        config = EmbeddingConfig(device="mps")
        config.validate()  # Should not raise

    def test_device_auto_detection(self):
        """Test that device is auto-detected when not specified."""
        config = EmbeddingConfig()
        # Device should be auto-detected to one of the valid options
        assert config.device in ["cpu", "cuda", "mps"]
        config.validate()  # Should not raise


@pytest.mark.unit
class TestSearchConfig:
    """Test SearchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SearchConfig()
        assert config.default_top_k == 5
        assert config.similarity_threshold is None
        assert config.index_type == "flat_ip"

    def test_validation_success(self):
        """Test successful validation."""
        config = SearchConfig(default_top_k=10, similarity_threshold=0.7)
        config.validate()  # Should not raise

    def test_validation_invalid_top_k(self):
        """Test validation fails for invalid top_k."""
        config = SearchConfig(default_top_k=0)
        with pytest.raises(ValueError, match="default_top_k must be >= 1"):
            config.validate()

    def test_validation_invalid_threshold_low(self):
        """Test validation fails for threshold < 0."""
        config = SearchConfig(similarity_threshold=-0.1)
        with pytest.raises(ValueError, match="must be between"):
            config.validate()

    def test_validation_invalid_threshold_high(self):
        """Test validation fails for threshold > 1."""
        config = SearchConfig(similarity_threshold=1.5)
        with pytest.raises(ValueError, match="must be between"):
            config.validate()

    def test_validation_invalid_index_type(self):
        """Test validation fails for invalid index type."""
        config = SearchConfig(index_type="invalid")
        with pytest.raises(ValueError, match="Invalid index_type"):
            config.validate()


@pytest.mark.unit
class TestRAGletConfigM2:
    """Test RAGletConfig with Milestone 2 extensions."""

    def test_default_config_includes_embedding_and_search(self):
        """Test default config includes embedding and search configs."""
        config = RAGletConfig()
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.search, SearchConfig)

    def test_validation_validates_nested_configs(self):
        """Test validation validates nested configs."""
        config = RAGletConfig()
        config.embedding.batch_size = 0  # Invalid
        with pytest.raises(ValueError):
            config.validate()

    def test_custom_nested_configs(self):
        """Test creating config with custom nested configs."""
        embedding_config = EmbeddingConfig(batch_size=64)
        search_config = SearchConfig(default_top_k=10)

        config = RAGletConfig(embedding=embedding_config, search=search_config)

        assert config.embedding.batch_size == 64
        assert config.search.default_top_k == 10
        config.validate()  # Should not raise
