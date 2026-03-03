"""Unit tests for configuration."""

import pytest

from raglet.config.config import ChunkingConfig, TinyRAGConfig


class TestChunkingConfig:
    """Test ChunkingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()

        assert config.size == 512
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
        config = ChunkingConfig(size=512, overlap=50)
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


class TestTinyRAGConfig:
    """Test TinyRAGConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TinyRAGConfig()

        assert isinstance(config.chunking, ChunkingConfig)
        assert config.chunking.size == 512
        assert config.custom_metadata == {}

    def test_custom_chunking_config(self):
        """Test custom chunking configuration."""
        chunking_config = ChunkingConfig(size=1024)
        config = TinyRAGConfig(chunking=chunking_config)

        assert config.chunking.size == 1024

    def test_custom_metadata(self):
        """Test custom metadata."""
        metadata = {"project": "test", "version": "1.0"}
        config = TinyRAGConfig(custom_metadata=metadata)

        assert config.custom_metadata == metadata

    def test_validate(self):
        """Test config validation."""
        config = TinyRAGConfig()
        config.validate()  # Should not raise

        # Test that invalid chunking config is caught
        config.chunking.size = 0
        with pytest.raises(ValueError):
            config.validate()
