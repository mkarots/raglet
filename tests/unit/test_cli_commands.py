"""Unit tests for CLI command functions."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raglet.cli import (
    add_command,
    build_command,
    package_command,
    query_command,
)
from raglet.core.chunk import Chunk


@pytest.mark.unit
class TestCLICommands:
    """Test CLI command functions."""

    def test_build_command_creates_raglet(self):
        """Test build_command creates raglet directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_path = workspace / "test.raglet"

            # Create test files
            (workspace / "test.txt").write_text("Test content")

            # Mock args
            args = MagicMock()
            args.inputs = [str(workspace / "test.txt")]
            args.out = str(output_path)
            args.ignore = ".git,__pycache__"
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 0
            assert output_path.exists()
            assert output_path.is_dir()
            assert (output_path / "chunks.json").exists()

    def test_build_command_handles_no_files(self):
        """Test build_command handles case with no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Mock args
            args = MagicMock()
            args.inputs = []  # No inputs
            args.out = None
            args.ignore = ""
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 1  # Should fail with no files

    def test_query_command_loads_and_searches(self):
        """Test query_command loads raglet and searches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "test.raglet.zip"

            # Create raglet
            chunks = [
                Chunk(text="Python programming", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.q = "Python"
            args.top_k = 5
            args.show_full = False

            # Run query command
            result = query_command(args)

            assert result == 0

    def test_query_command_handles_missing_raglet(self):
        """Test query_command handles missing raglet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "nonexistent.raglet.zip"

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.q = "test"
            args.top_k = 5
            args.show_full = False

            # Run query command (should fail)
            result = query_command(args)

            assert result == 1  # Should fail with missing raglet

    def test_add_command_adds_files(self):
        """Test add_command adds files to raglet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "test.raglet.zip"

            # Create initial raglet
            chunks = [
                Chunk(text="Initial", source="initial.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Create new file
            new_file = workspace / "new.txt"
            new_file.write_text("New content")

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.files = [str(new_file)]
            args.out = None

            # Run add command
            result = add_command(args)

            assert result == 0

            # Verify file added
            loaded = RAGlet.load(str(kb_path))
            assert len(loaded.chunks) > len(chunks)

    def test_add_command_handles_missing_raglet(self):
        """Test add_command handles missing raglet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "nonexistent.raglet.zip"

            # Create test file
            test_file = workspace / "test.txt"
            test_file.write_text("Test content")

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.files = [str(test_file)]
            args.out = None

            # Run add command (should fail)
            result = add_command(args)

            assert result == 1  # Should fail with missing raglet

    def test_package_command_creates_zip(self):
        """Test package_command creates zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"
            zip_path = workspace / "export.zip"

            # Create knowledge base
            chunks = [
                Chunk(text="Test", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.format = "zip"
            args.out = str(zip_path)

            # Run package command
            result = package_command(args)

            assert result == 0
            assert zip_path.exists()

    def test_package_command_handles_missing_raglet(self):
        """Test package_command handles missing raglet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "nonexistent.raglet"

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.format = "zip"
            args.out = None

            # Run package command (should fail)
            result = package_command(args)

            assert result == 1  # Should fail with missing raglet

    def test_build_command_with_custom_output(self):
        """Test build_command with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_path = workspace / "custom_output.raglet"

            # Create test files
            (workspace / "test.txt").write_text("Test content")

            # Mock args
            args = MagicMock()
            args.inputs = [str(workspace / "test.txt")]
            args.out = str(output_path)
            args.ignore = ""
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 0
            assert output_path.exists()
            assert output_path.is_dir()

    def test_query_command_with_custom_raglet_path(self):
        """Test query_command with custom raglet path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "custom.raglet.zip"

            # Create raglet
            chunks = [
                Chunk(text="Python", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.q = "Python"
            args.top_k = 5
            args.show_full = False

            # Run query command
            result = query_command(args)

            assert result == 0
