"""End-to-end tests for raglet CLI."""

import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.e2e
class TestCLIE2E:
    """End-to-end tests for raglet CLI."""

    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            ["raglet", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "raglet" in result.stdout.lower()
        assert "build" in result.stdout.lower()
        assert "query" in result.stdout.lower()

    def test_cli_build_query_workflow(self):
        """Test complete build → query workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"

            # Create test files
            (workspace / "doc1.txt").write_text("Python is a programming language.")
            (workspace / "doc2.md").write_text("# Machine Learning\n\nML uses algorithms.")

            # Build knowledge base using Python API (simulating CLI)
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files(
                [
                    str(workspace / "doc1.txt"),
                    str(workspace / "doc2.md"),
                ],
                config=config,
            )
            raglet.save(str(kb_path))

            # Verify knowledge base created
            assert kb_path.exists()

            # Query using Python API (simulating CLI)
            loaded = RAGlet.load(str(kb_path))
            results = loaded.search("Python", top_k=1)

            assert len(results) > 0
            assert "python" in results[0].text.lower()

    def test_cli_incremental_add_workflow(self):
        """Test incremental add workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"

            # Create initial knowledge base
            from raglet import RAGlet, RAGletConfig
            from raglet.core.chunk import Chunk

            chunks = [
                Chunk(text="Initial content", source="initial.txt", index=0),
            ]
            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Create new file
            new_file = workspace / "new.txt"
            new_file.write_text("New content to add.")

            # Add file incrementally (simulating CLI)
            loaded = RAGlet.load(str(kb_path))
            loaded.add_files([str(new_file)])
            loaded.save(str(kb_path), incremental=True)

            # Verify incremental add worked
            reloaded = RAGlet.load(str(kb_path))
            assert len(reloaded.chunks) > len(chunks)
            assert any("New content" in chunk.text for chunk in reloaded.chunks)

    def test_cli_export_workflow(self):
        """Test export workflow."""
        import zipfile

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"
            zip_path = workspace / "export.zip"

            # Create knowledge base
            from raglet import RAGlet, RAGletConfig
            from raglet.core.chunk import Chunk

            chunks = [
                Chunk(text="Export test", source="test.txt", index=0),
            ]
            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Export to zip (simulating CLI)
            with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_name in ["config.json", "chunks.json", "embeddings.npy", "metadata.json"]:
                    file_path = kb_path / file_name
                    if file_path.exists():
                        zipf.write(file_path, file_name)

            # Verify export
            assert zip_path.exists()
            with zipfile.ZipFile(str(zip_path), "r") as zipf:
                files = zipf.namelist()
                assert "config.json" in files
                assert "chunks.json" in files

    def test_cli_multi_session_workflow(self):
        """Test multi-session workflow (simulating agent memory use case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"

            # Session 1: Create initial memory
            from raglet import RAGlet, RAGletConfig
            from raglet.core.chunk import Chunk

            chunks1 = [
                Chunk(text="Session 1: User said hello", source="session1.txt", index=0),
                Chunk(text="Session 1: Assistant responded", source="session1.txt", index=1),
            ]
            config = RAGletConfig()
            raglet1 = RAGlet.from_files([], config=config)
            raglet1.chunks = chunks1
            raglet1.embeddings = raglet1.embedding_generator.generate(chunks1)
            raglet1.vector_store.add_vectors(raglet1.embeddings, chunks1)
            raglet1.save(str(kb_path))

            # Session 2: Add more memory
            loaded = RAGlet.load(str(kb_path))
            chunks2 = [
                Chunk(text="Session 2: User asked about Python", source="session2.txt", index=2),
                Chunk(text="Session 2: Assistant explained Python", source="session2.txt", index=3),
            ]
            loaded.add_chunks(chunks2)
            loaded.save(str(kb_path), incremental=True)

            # Session 3: Load and search
            reloaded = RAGlet.load(str(kb_path))
            assert len(reloaded.chunks) == 4

            # Search across all sessions
            results = reloaded.search("Python", top_k=2)
            assert len(results) > 0
            assert any("Python" in r.text for r in results)

    def test_cli_workspace_not_found(self):
        """Test CLI handles missing workspace gracefully."""
        # This would be tested with actual CLI invocation
        # For now, test the underlying logic
        from unittest.mock import MagicMock

        from raglet.cli import build_command

        args = MagicMock()
        args.workspace = "/nonexistent/directory"
        args.kb_name = None
        args.extensions = ".txt"
        args.ignore = ""
        args.max_files = None
        args.chunk_size = None
        args.chunk_overlap = None
        args.model = None

        result = build_command(args)
        assert result == 1  # Should fail gracefully
