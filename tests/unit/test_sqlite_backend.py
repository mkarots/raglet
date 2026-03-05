"""Unit tests for SQLiteStorageBackend."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from raglet.config.config import RAGletConfig
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet
from raglet.storage.sqlite_backend import SQLiteStorageBackend


@pytest.mark.unit
class TestSQLiteStorageBackend:
    """Test SQLiteStorageBackend."""

    def test_supports_incremental(self):
        """Test supports_incremental() returns True."""
        backend = SQLiteStorageBackend()
        assert backend.supports_incremental() is True

    def test_save_and_load_empty_raglet(self):
        """Test saving and loading empty RAGlet."""
        backend = SQLiteStorageBackend()
        config = RAGletConfig()
        raglet = RAGlet(chunks=[], config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.sqlite"
            backend.save(raglet, str(file_path))

            loaded = backend.load(str(file_path))
            assert len(loaded.chunks) == 0
            assert loaded.config.chunking.size == config.chunking.size

    def test_save_and_load_with_chunks(self):
        """Test saving and loading RAGlet with chunks."""
        backend = SQLiteStorageBackend()
        config = RAGletConfig()
        chunks = [
            Chunk(text="Test chunk 1", source="test.txt", index=0),
            Chunk(text="Test chunk 2", source="test.txt", index=1),
        ]
        raglet = RAGlet(chunks=chunks, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.sqlite"
            backend.save(raglet, str(file_path))

            loaded = backend.load(str(file_path))
            assert len(loaded.chunks) == 2
            assert loaded.chunks[0].text == "Test chunk 1"
            assert loaded.chunks[1].text == "Test chunk 2"
            assert loaded.chunks[0].source == "test.txt"

    def test_save_and_load_preserves_embeddings(self):
        """Test that embeddings are preserved after save/load."""
        backend = SQLiteStorageBackend()
        config = RAGletConfig()
        chunks = [
            Chunk(text="Test chunk 1", source="test.txt", index=0),
            Chunk(text="Test chunk 2", source="test.txt", index=1),
        ]
        raglet = RAGlet(chunks=chunks, config=config)

        original_embeddings = raglet.embeddings.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.sqlite"
            backend.save(raglet, str(file_path))

            loaded = backend.load(str(file_path))
            assert loaded.embeddings.shape == original_embeddings.shape
            np.testing.assert_array_almost_equal(loaded.embeddings, original_embeddings, decimal=5)

    def test_save_and_load_preserves_config(self):
        """Test that config is preserved after save/load."""
        backend = SQLiteStorageBackend()
        config = RAGletConfig()
        config.chunking.size = 1024
        config.embedding.model = "all-MiniLM-L6-v2"
        config.search.default_top_k = 10
        config.custom_metadata = {"project": "test"}

        raglet = RAGlet(chunks=[], config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.sqlite"
            backend.save(raglet, str(file_path))

            loaded = backend.load(str(file_path))
            assert loaded.config.chunking.size == 1024
            assert loaded.config.embedding.model == "all-MiniLM-L6-v2"
            assert loaded.config.search.default_top_k == 10
            assert loaded.config.custom_metadata == {"project": "test"}

    def test_incremental_save(self):
        """Test incremental save appends new chunks."""
        backend = SQLiteStorageBackend()
        config = RAGletConfig()
        chunks1 = [Chunk(text="Chunk 1", source="test.txt", index=0)]
        raglet1 = RAGlet(chunks=chunks1, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.sqlite"
            backend.save(raglet1, str(file_path))

            # Add more chunks
            chunks2 = [
                Chunk(text="Chunk 1", source="test.txt", index=0),
                Chunk(text="Chunk 2", source="test.txt", index=1),
            ]
            raglet2 = RAGlet(chunks=chunks2, config=config)
            backend.save(raglet2, str(file_path), incremental=True)

            loaded = backend.load(str(file_path))
            assert len(loaded.chunks) == 2
            assert loaded.chunks[0].text == "Chunk 1"
            assert loaded.chunks[1].text == "Chunk 2"

    def test_add_chunks(self):
        """Test add_chunks() method."""
        backend = SQLiteStorageBackend()
        config = RAGletConfig()
        chunks = [Chunk(text="Chunk 1", source="test.txt", index=0)]
        raglet = RAGlet(chunks=chunks, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.sqlite"
            backend.save(raglet, str(file_path))

            # Add new chunks
            new_chunks = [Chunk(text="Chunk 2", source="test.txt", index=1)]
            new_embeddings = raglet.embedding_generator.generate(new_chunks)
            backend.add_chunks(str(file_path), new_chunks, new_embeddings)

            loaded = backend.load(str(file_path))
            assert len(loaded.chunks) == 2
            assert loaded.chunks[0].text == "Chunk 1"
            assert loaded.chunks[1].text == "Chunk 2"

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        backend = SQLiteStorageBackend()
        with pytest.raises(FileNotFoundError):
            backend.load("nonexistent.sqlite")

    def test_save_creates_directory(self):
        """Test that save() creates parent directory if needed."""
        backend = SQLiteStorageBackend()
        config = RAGletConfig()
        raglet = RAGlet(chunks=[], config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test.sqlite"
            backend.save(raglet, str(file_path))

            assert file_path.exists()

    def test_full_save_replaces_data(self):
        """Test that full save (not incremental) replaces all data."""
        backend = SQLiteStorageBackend()
        config = RAGletConfig()
        chunks1 = [
            Chunk(text="Chunk 1", source="test.txt", index=0),
            Chunk(text="Chunk 2", source="test.txt", index=1),
        ]
        raglet1 = RAGlet(chunks=chunks1, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.sqlite"
            backend.save(raglet1, str(file_path))

            # Full save with different chunks
            chunks2 = [Chunk(text="New chunk", source="test2.txt", index=0)]
            raglet2 = RAGlet(chunks=chunks2, config=config)
            backend.save(raglet2, str(file_path), incremental=False)

            loaded = backend.load(str(file_path))
            assert len(loaded.chunks) == 1
            assert loaded.chunks[0].text == "New chunk"
            assert loaded.chunks[0].source == "test2.txt"
