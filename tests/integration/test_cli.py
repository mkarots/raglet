"""Integration tests for raglet CLI."""

import tempfile
from pathlib import Path

import pytest

from raglet import RAGlet, RAGletConfig
from raglet.core.chunk import Chunk


@pytest.mark.integration
class TestCLI:
    """Test raglet CLI commands."""

    def test_build_command_creates_knowledge_base(self):
        """Test 'build' command creates knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            # Create test files
            (workspace / "test1.txt").write_text("Python is a programming language.")
            (workspace / "test2.md").write_text("# Machine Learning\n\nML uses algorithms.")

            # Build knowledge base
            config = RAGletConfig()
            raglet = RAGlet.from_files(
                [str(workspace / "test1.txt"), str(workspace / "test2.md")],
                config=config,
            )
            raglet.save(str(kb_path))

            # Verify knowledge base exists
            assert kb_path.exists()
            assert (kb_path / "config.json").exists()
            assert (kb_path / "chunks.json").exists()
            assert (kb_path / "embeddings.npy").exists()
            assert (kb_path / "metadata.json").exists()

            # Verify chunks
            loaded = RAGlet.load(str(kb_path))
            assert len(loaded.chunks) > 0

    def test_query_command_returns_results(self):
        """Test 'query' command returns search results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            # Create and save knowledge base
            chunks = [
                Chunk(text="Python programming language", source="test.txt", index=0),
                Chunk(text="Machine learning algorithms", source="test.txt", index=1),
            ]
            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Query
            loaded = RAGlet.load(str(kb_path))
            results = loaded.search("Python", top_k=1)

            assert len(results) > 0
            assert "python" in results[0].text.lower()

    def test_add_command_adds_files(self):
        """Test 'add' command adds files incrementally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            # Create initial knowledge base
            chunks = [
                Chunk(text="Initial chunk", source="initial.txt", index=0),
            ]
            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Create new file
            new_file = workspace / "new_file.txt"
            new_file.write_text("New content to add.")

            # Add file
            loaded = RAGlet.load(str(kb_path))
            loaded.add_files([str(new_file)])
            loaded.save(str(kb_path), incremental=True)

            # Verify new chunks added
            reloaded = RAGlet.load(str(kb_path))
            assert len(reloaded.chunks) > len(chunks)
            assert any("New content" in chunk.text for chunk in reloaded.chunks)

    def test_export_command_creates_zip(self):
        """Test 'export' command creates zip file."""
        import zipfile

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"
            zip_path = workspace / "export.zip"

            # Create knowledge base
            chunks = [
                Chunk(text="Test content", source="test.txt", index=0),
            ]
            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Export to zip
            with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_name in ["config.json", "chunks.json", "embeddings.npy", "metadata.json"]:
                    file_path = kb_path / file_name
                    if file_path.exists():
                        zipf.write(file_path, file_name)

            # Verify zip exists and contains files
            assert zip_path.exists()
            with zipfile.ZipFile(str(zip_path), "r") as zipf:
                files = zipf.namelist()
                assert "config.json" in files
                assert "chunks.json" in files
                assert "embeddings.npy" in files
                assert "metadata.json" in files

    def test_inspect_command_shows_info(self):
        """Test 'inspect' command shows knowledge base information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            # Create knowledge base
            chunks = [
                Chunk(text="Python programming", source="test1.txt", index=0),
                Chunk(text="Machine learning", source="test2.txt", index=1),
            ]
            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Inspect
            loaded = RAGlet.load(str(kb_path))

            # Verify we can access all info
            assert len(loaded.chunks) == 2
            assert loaded.config.embedding.model == "all-MiniLM-L6-v2"
            assert loaded.embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

            # Check sources
            sources = {chunk.source for chunk in loaded.chunks}
            assert len(sources) == 2
            assert "test1.txt" in sources
            assert "test2.txt" in sources

    def test_build_with_custom_config(self):
        """Test 'build' command with custom chunk size and model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            # Create test file
            (workspace / "test.txt").write_text("Python is a programming language. " * 100)

            # Build with custom config
            from raglet.config.config import ChunkingConfig, EmbeddingConfig

            config = RAGletConfig(
                chunking=ChunkingConfig(size=100, overlap=10),
                embedding=EmbeddingConfig(model="all-MiniLM-L6-v2"),
            )
            raglet = RAGlet.from_files([str(workspace / "test.txt")], config=config)
            raglet.save(str(kb_path))

            # Verify config saved
            loaded = RAGlet.load(str(kb_path))
            assert loaded.config.chunking.size == 100
            assert loaded.config.chunking.overlap == 10
            assert loaded.config.embedding.model == "all-MiniLM-L6-v2"

    def test_query_with_top_k(self):
        """Test 'query' command respects top_k parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            # Create knowledge base with multiple chunks
            chunks = [
                Chunk(text=f"Chunk {i}: Python programming", source="test.txt", index=i)
                for i in range(10)
            ]
            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Query with top_k=3
            loaded = RAGlet.load(str(kb_path))
            results = loaded.search("Python", top_k=3)

            assert len(results) == 3

    def test_incremental_save_preserves_existing_chunks(self):
        """Test incremental save preserves existing chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            # Create initial knowledge base
            chunks1 = [
                Chunk(text="Initial chunk 1", source="initial.txt", index=0),
                Chunk(text="Initial chunk 2", source="initial.txt", index=1),
            ]
            config = RAGletConfig()
            raglet1 = RAGlet.from_files([], config=config)
            raglet1.chunks = chunks1
            raglet1.embeddings = raglet1.embedding_generator.generate(chunks1)
            raglet1.vector_store.add_vectors(raglet1.embeddings, chunks1)
            raglet1.save(str(kb_path))

            # Add new chunks incrementally
            loaded = RAGlet.load(str(kb_path))
            chunks2 = [
                Chunk(text="New chunk 1", source="new.txt", index=2),
                Chunk(text="New chunk 2", source="new.txt", index=3),
            ]
            loaded.add_chunks(chunks2)
            loaded.save(str(kb_path), incremental=True)

            # Verify all chunks present
            reloaded = RAGlet.load(str(kb_path))
            assert len(reloaded.chunks) == 4
            assert reloaded.chunks[0].text == "Initial chunk 1"
            assert reloaded.chunks[3].text == "New chunk 2"

    def test_build_ignores_patterns(self):
        """Test 'build' command ignores specified patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            # Create files
            (workspace / "include.txt").write_text("Include this.")
            (workspace / ".git").mkdir(parents=True, exist_ok=True)
            (workspace / ".git" / "ignore.txt").write_text("Ignore this.")
            (workspace / "__pycache__").mkdir(parents=True, exist_ok=True)
            (workspace / "__pycache__" / "ignore.pyc").write_bytes(b"binary")

            # Build (should only include include.txt)
            config = RAGletConfig()
            files_to_process = [
                str(workspace / "include.txt"),
            ]
            raglet = RAGlet.from_files(files_to_process, config=config)
            raglet.save(str(kb_path))

            # Verify only included file processed
            loaded = RAGlet.load(str(kb_path))
            sources = {chunk.source for chunk in loaded.chunks}
            assert str(workspace / "include.txt") in sources
            assert str(workspace / ".git" / "ignore.txt") not in sources
