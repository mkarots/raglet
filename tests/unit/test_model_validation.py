"""Unit tests for model validation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from raglet.config.config import EmbeddingConfig, RAGletConfig
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet
from raglet.embeddings.generator import SentenceTransformerGenerator


@pytest.mark.unit
class TestModelValidation:
    """Test model validation and dimension checking."""

    def test_dimension_mismatch_on_init(self):
        """Test that providing embeddings with wrong dimension raises error."""
        config = RAGletConfig(embedding=EmbeddingConfig(model="all-MiniLM-L6-v2"))  # 384 dims
        chunks = [
            Chunk(text="Test chunk", source="test.txt", index=0),
        ]

        # Create wrong-dimension embeddings (768 instead of 384)
        wrong_embeddings = np.random.rand(1, 768).astype(np.float32)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            RAGlet(
                chunks=chunks,
                config=config,
                embeddings=wrong_embeddings,
            )

    def test_dimension_mismatch_on_search(self):
        """Test that search fails if model dimension doesn't match stored embeddings."""
        # Create RAGlet with one model
        config1 = RAGletConfig(embedding=EmbeddingConfig(model="all-MiniLM-L6-v2"))  # 384 dims
        chunks = [
            Chunk(text="Test chunk", source="test.txt", index=0),
        ]
        raglet = RAGlet.from_files([], config=config1)
        raglet.chunks = chunks
        raglet.embeddings = np.random.rand(1, 384).astype(np.float32)
        raglet.vector_store.add_vectors(raglet.embeddings, chunks)

        # Change to different model (different dimension)
        config2 = RAGletConfig(embedding=EmbeddingConfig(model="all-mpnet-base-v2"))  # 768 dims
        raglet.config = config2
        raglet.embedding_generator = SentenceTransformerGenerator(config2.embedding)

        # Search should fail
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            raglet.search("test query")

    def test_load_with_dimension_mismatch(self):
        """Test that loading fails if saved embeddings don't match model dimension."""
        # Create and save RAGlet
        config = RAGletConfig(embedding=EmbeddingConfig(model="all-MiniLM-L6-v2"))  # 384 dims
        chunks = [
            Chunk(text="Test chunk", source="test.txt", index=0),
        ]
        raglet = RAGlet.from_files([], config=config)
        raglet.chunks = chunks
        raglet.embeddings = np.random.rand(1, 384).astype(np.float32)
        raglet.vector_store.add_vectors(raglet.embeddings, chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(file_path))

            # Manually corrupt the config in the database to simulate model change
            import json
            import sqlite3

            conn = sqlite3.connect(str(file_path))
            # Change model to one with different dimension
            config_dict = json.loads(
                conn.execute("SELECT value FROM metadata WHERE key = 'config'").fetchone()[0]
            )
            config_dict["embedding"]["model"] = "all-mpnet-base-v2"  # 768 dims
            conn.execute(
                "UPDATE metadata SET value = ? WHERE key = ?", (json.dumps(config_dict), "config")
            )
            conn.commit()
            conn.close()

            # Loading should fail
            with pytest.raises(ValueError, match="Embedding dimension mismatch"):
                RAGlet.load(str(file_path))

    def test_correct_dimension_works(self):
        """Test that correct dimension works fine."""
        config = RAGletConfig(embedding=EmbeddingConfig(model="all-MiniLM-L6-v2"))  # 384 dims
        chunks = [
            Chunk(text="Test chunk", source="test.txt", index=0),
        ]

        # Create correct-dimension embeddings
        correct_embeddings = np.random.rand(1, 384).astype(np.float32)

        raglet = RAGlet(
            chunks=chunks,
            config=config,
            embeddings=correct_embeddings,
        )

        # Should work fine
        assert raglet.embeddings.shape == (1, 384)
