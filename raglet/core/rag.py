"""RAGlet main class."""

import atexit
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from raglet.config.config import RAGletConfig
from raglet.core.chunk import Chunk
from raglet.embeddings.interfaces import EmbeddingGenerator
from raglet.processing.interfaces import Chunker, DocumentExtractor
from raglet.vector_store.interfaces import VectorStore

if TYPE_CHECKING:
    from raglet.cli_utils import CLIOutput
    from raglet.storage.interfaces import StorageBackend

# Default ignore patterns for file discovery
# These patterns are excluded when processing directories or glob patterns
DEFAULT_IGNORE_PATTERNS = [".git", "__pycache__", ".venv", "node_modules", ".raglet", "assets", "*.egg-info", "*.pyc", "*.pyo", "*.pyd", "*.pyw", "*.pyz"]


class RAGlet:
    """Main RAG orchestrator class."""

    def __init__(
        self,
        chunks: list[Chunk],
        config: RAGletConfig,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        embeddings: Optional[np.ndarray] = None,
        auto_save_path: Optional[str] = None,
        auto_save_threshold: Optional[int] = None,
    ):
        """Initialize RAGlet.

        Args:
            chunks: List of chunks
            config: Configuration
            embedding_generator: Optional embedding generator (created if None)
            vector_store: Optional vector store (created if None)
            embeddings: Optional pre-computed embeddings (generated if None and chunks exist)
            auto_save_path: Optional path for automatic saves (enables buffering)
            auto_save_threshold: Optional threshold in characters for auto-save (default: 1000)
                                 Only used if auto_save_path is set
        """
        self.chunks = chunks
        self.config = config

        # Auto-save configuration
        self._auto_save_path: Optional[str] = auto_save_path
        self._auto_save_threshold: int = auto_save_threshold if auto_save_threshold is not None else 1000
        self._unsaved_chars: int = 0

        # Register exit handler if auto-save enabled
        if self._auto_save_path:
            atexit.register(self._save_on_exit)

        # Create default implementations if not provided
        if embedding_generator is None:
            from raglet.embeddings.generator import SentenceTransformerGenerator

            self.embedding_generator: EmbeddingGenerator = SentenceTransformerGenerator(
                config.embedding
            )
        else:
            self.embedding_generator = embedding_generator

        if vector_store is None:
            from raglet.vector_store.faiss_store import FAISSVectorStore

            self.vector_store: VectorStore = FAISSVectorStore(
                embedding_dim=self.embedding_generator.get_dimension(),
                config=config.search,
            )
        else:
            self.vector_store = vector_store

        # Generate or use provided embeddings
        if chunks:
            if embeddings is not None:
                # Validate provided embeddings match generator dimension
                provided_dim = embeddings.shape[1] if len(embeddings) > 0 else 0
                generator_dim = self.embedding_generator.get_dimension()

                if provided_dim != generator_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: Provided embeddings have dimension {provided_dim}, "
                        f"but model '{self.config.embedding.model}' produces dimension {generator_dim}. "
                        f"Embeddings must match the model's dimension."
                    )

                self.embeddings = embeddings
            else:
                self.embeddings = self.embedding_generator.generate(chunks)
            self.vector_store.add_vectors(self.embeddings, chunks)
        else:
            self.embeddings = np.array([]).reshape(0, self.embedding_generator.get_dimension())

    @classmethod
    def from_files(
        cls,
        files: list[str],
        document_extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[Chunker] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[RAGletConfig] = None,
        ignore_patterns: Optional[list[str]] = None,
        output: Optional["CLIOutput"] = None,
    ) -> "RAGlet":
        """Create RAGlet from files, directories, or glob patterns.

        Full pipeline: Expand inputs → Extract → Chunk → Embed → Index

        Supports:
        - Individual files: `["file.txt"]`
        - Directories: `["docs/"]` (recursively finds all files)
        - Glob patterns: `["*.py"]`, `["**/*.md"]`, `["docs/**/*.txt"]`

        Args:
            files: List of file paths, directory paths, or glob patterns
            document_extractor: Optional document extractor (uses factory if None)
            chunker: Optional chunker (creates default if None)
            embedding_generator: Optional embedding generator (creates default if None)
            vector_store: Optional vector store (creates default if None)
            config: Optional configuration (uses defaults if None)
            ignore_patterns: Optional list of patterns to ignore (e.g., [".git", "__pycache__"])
            output: Optional CLI output handler for progress messages

        Returns:
            RAGlet instance with searchable index

        Raises:
            FileNotFoundError: If any file doesn't exist
            ValueError: If configuration is invalid or extraction fails
        """
        if config is None:
            config = RAGletConfig()

        config.validate()

        # Expand inputs: files, directories, glob patterns
        from raglet.utils import expand_file_inputs

        # Handle empty file list (allows creating empty RAGlet)
        if not files:
            filtered_files = []
        else:
            filtered_files = expand_file_inputs(files, ignore_patterns=ignore_patterns)

        # Step 1: Extract text from files
        from raglet.processing.chunker import SentenceAwareChunker
        from raglet.processing.extractor_factory import create_extractor

        if output and filtered_files:
            output.progress(f"Extracting text from {len(filtered_files)} file{'s' if len(filtered_files) != 1 else ''}...")

        all_chunks = []
        for i, file_path in enumerate(filtered_files, 1):
            if output and i % max(1, len(filtered_files) // 10) == 0:
                output.verbose_msg(f"  Processing file {i}/{len(filtered_files)}: {Path(file_path).name}")

            if document_extractor is None:
                extractor = create_extractor(file_path)
            else:
                extractor = document_extractor

            text = extractor.extract(file_path)

            # Step 2: Chunk text
            if chunker is None:
                chunker_instance: Chunker = SentenceAwareChunker(config.chunking)
            else:
                chunker_instance = chunker

            chunks = chunker_instance.chunk(text, metadata={"source": file_path})
            all_chunks.extend(chunks)

        if output:
            output.progress(f"Chunked text into {len(all_chunks)} chunk{'s' if len(all_chunks) != 1 else ''}...")
            output.progress("Generating embeddings...")

        # Step 3 & 4: Generate embeddings and create vector store
        # (handled in __init__)
        raglet = cls(
            chunks=all_chunks,
            config=config,
            embedding_generator=embedding_generator,
            vector_store=vector_store,
        )

        if output:
            output.progress("Indexing vectors...")

        return raglet

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[Chunk]:
        """Search and retrieve relevant chunks.

        Args:
            query: Search query string
            top_k: Number of results (uses config default if None)
            similarity_threshold: Minimum similarity score (uses config default if None)

        Returns:
            List of Chunk objects with score attribute set, sorted by similarity
            (most similar first). Returns empty list if no chunks or threshold not met.
        """
        if not self.chunks:
            return []

        top_k = top_k or self.config.search.default_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.config.search.similarity_threshold
        )

        # Validate query embedding dimension matches stored embeddings
        query_embedding = self.embedding_generator.generate_single(query)

        if len(self.embeddings) > 0:
            stored_dim = self.embeddings.shape[1]
            query_dim = query_embedding.shape[0]

            if stored_dim != query_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: Stored embeddings have dimension {stored_dim}, "
                    f"but query embedding has dimension {query_dim}. "
                    f"This indicates the embedding model has changed. "
                    f"Current model: '{self.config.embedding.model}'. "
                    f"To fix: Regenerate embeddings with the current model or use the model that matches "
                    f"the stored embeddings."
                )

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Apply similarity threshold if configured
        # Cosine similarity: scores are 0-1 range, threshold is 0-1 (higher = more similar)
        if threshold is not None:
            results = [r for r in results if r.score is not None and r.score >= threshold]

        return results

    def get_all_chunks(self) -> list[Chunk]:
        """Get all chunks.

        Returns:
            List of all chunks
        """
        return self.chunks

    def save(
        self,
        file_path: str,
        incremental: bool = False,
        storage_backend: Optional["StorageBackend"] = None,
    ) -> None:
        """Save RAGlet to storage (directory, SQLite file, or zip archive).

        Auto-detects format:
        - If path ends with .sqlite or .db → SQLite format
        - If path ends with .zip → Zip format (read-only, no incremental updates)
        - If path is directory → Directory format
        - Otherwise → Directory format (default)

        Args:
            file_path: Path to output directory or .sqlite file
            incremental: If True, append new chunks (if backend supports it)
            storage_backend: Optional storage backend (auto-detects if None)

        Raises:
            ValueError: If serialization fails
            IOError: If file operations fail
        """
        backend = storage_backend or self._get_default_backend(file_path)
        backend.save(self, file_path, incremental=incremental)

    @classmethod
    def load(
        cls,
        file_path: str,
        storage_backend: Optional["StorageBackend"] = None,
    ) -> "RAGlet":
        """Load RAGlet from storage (directory, SQLite file, or zip archive).

        Auto-detects format:
        - If path ends with .sqlite or .db → SQLite format
        - If path ends with .zip → Zip format
        - If path is directory → Directory format
        - Otherwise → Directory format (default)

        Args:
            file_path: Path to directory or .sqlite file
            storage_backend: Optional storage backend (auto-detects if None)

        Returns:
            RAGlet instance

        Raises:
            FileNotFoundError: If file/directory doesn't exist
            ValueError: If file format is invalid
            IOError: If file operations fail
        """
        backend = storage_backend or cls._detect_backend(file_path)
        return backend.load(file_path)

    def add_files(
        self,
        files: list[str],
        document_extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[Chunker] = None,
        file_path: Optional[str] = None,
        output: Optional["CLIOutput"] = None,
    ) -> None:
        """Add files to existing RAGlet.

        Extracts, chunks, and embeds new files, then adds them to the existing RAGlet.

        Args:
            files: List of file paths to add
            document_extractor: Optional document extractor (uses factory if None)
            chunker: Optional chunker (uses RAGlet's config if None)
            file_path: Optional file path (if provided, saves immediately)
            output: Optional CLI output handler for progress messages

        Raises:
            FileNotFoundError: If any file doesn't exist
            ValueError: If extraction fails
        """
        if not files:
            return

        if output:
            output.progress(f"Extracting text from {len(files)} file{'s' if len(files) != 1 else ''}...")

        # Extract and chunk files (similar to from_files logic)
        from raglet.processing.chunker import SentenceAwareChunker
        from raglet.processing.extractor_factory import create_extractor

        new_chunks = []

        for file_path_item in files:
            if output:
                output.verbose_msg(f"  Processing: {Path(file_path_item).name}")

            if document_extractor is None:
                extractor = create_extractor(file_path_item)
            else:
                extractor = document_extractor

            text = extractor.extract(file_path_item)

            # Chunk text
            if chunker is None:
                chunker_instance: Chunker = SentenceAwareChunker(self.config.chunking)
            else:
                chunker_instance = chunker

            chunks = chunker_instance.chunk(text, metadata={"source": file_path_item})
            new_chunks.extend(chunks)

        if output:
            output.progress(f"Chunked into {len(new_chunks)} chunk{'s' if len(new_chunks) != 1 else ''}...")
            output.progress("Generating embeddings...")

        # Add chunks (reuse add_chunks logic)
        if new_chunks:
            self.add_chunks(new_chunks, file_path=file_path)

    def add_text(
        self,
        text: str,
        source: str = "manual",
        metadata: Optional[dict[str, Any]] = None,
        file_path: Optional[str] = None,
    ) -> None:
        """Add raw text to RAGlet (chunks automatically).

        Convenience method for adding raw text. The text will be chunked using
        the RAGlet's chunking configuration.

        Args:
            text: Raw text to add
            source: Source identifier for the text (default: "manual")
            metadata: Optional metadata dictionary
            file_path: Optional file path (if provided, saves immediately)

        Raises:
            ValueError: If text is empty
        """
        if not text:
            return

        from raglet.processing.chunker import SentenceAwareChunker

        # Chunk text using RAGlet's chunking config
        chunker = SentenceAwareChunker(self.config.chunking)
        chunk_metadata = metadata or {}
        chunk_metadata["source"] = source

        chunks = chunker.chunk(text, metadata=chunk_metadata)
        self.add_chunks(chunks, file_path=file_path)

    def add_file(
        self,
        file_path_item: str,
        document_extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[Chunker] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Add a single file to RAGlet.

        Convenience method for adding a single file. Equivalent to calling
        add_files with a single-item list.

        Args:
            file_path_item: Path to file to add
            document_extractor: Optional document extractor (uses factory if None)
            chunker: Optional chunker (uses RAGlet's config if None)
            save_path: Optional file path (if provided, saves immediately)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If extraction fails
        """
        self.add_files([file_path_item], document_extractor, chunker, save_path)

    def add_chunks(
        self,
        new_chunks: list[Chunk],
        file_path: Optional[str] = None,
    ) -> None:
        """Add chunks incrementally.

        Chunk indices are automatically assigned based on the current number of chunks.
        If chunks already have indices set, they will be reassigned to ensure continuity.

        Args:
            new_chunks: New chunks to add (indices will be auto-assigned)
            file_path: Optional file path (if provided, saves immediately)
        """
        if not new_chunks:
            return

        # Auto-assign chunk indices based on current chunk count
        current_index = len(self.chunks)
        for i, chunk in enumerate(new_chunks):
            # Reassign index to ensure continuity
            chunk.index = current_index + i

        # Generate embeddings for new chunks
        new_embeddings = self.embedding_generator.generate(new_chunks)

        # Add to vector store
        self.vector_store.add_vectors(new_embeddings, new_chunks)

        # Update in-memory state
        self.chunks.extend(new_chunks)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Save if file path provided
        if file_path:
            backend = self._get_default_backend(file_path)
            if backend.supports_incremental():
                backend.add_chunks(file_path, new_chunks, new_embeddings, self)
            else:
                # Fallback to full save
                self.save(file_path, incremental=False, storage_backend=backend)

    @classmethod
    def from_sqlite(
        cls,
        db_path: str,
        storage_backend: Optional["StorageBackend"] = None,
    ) -> "RAGlet":
        """Load RAGlet from existing SQLite database.

        Useful for inspection, analysis, or converting existing databases.

        Args:
            db_path: Path to SQLite database file
            storage_backend: Optional storage backend (auto-detects if None)

        Returns:
            RAGlet instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
            IOError: If file operations fail
        """
        backend = storage_backend or cls._detect_backend(db_path)
        return backend.load(db_path)

    @staticmethod
    def _detect_backend(file_path: str) -> "StorageBackend":
        """Auto-detect storage backend from file path.

        Detection logic:
        1. If path has .sqlite or .db extension → SQLite format
        2. If path has .zip extension → Zip format
        3. If path exists and is a directory → Directory format
        4. If path exists and is a file with SQLite magic bytes → SQLite format
        5. If path exists and is a file with Zip magic bytes (PK) → Zip format
        6. Otherwise → Directory format (default)

        Args:
            file_path: Path to file or directory

        Returns:
            StorageBackend instance
        """
        path = Path(file_path)

        # SQLite format (by extension)
        if path.suffix in [".sqlite", ".db"]:
            from raglet.storage.sqlite_backend import SQLiteStorageBackend

            return SQLiteStorageBackend()

        # Zip format (by extension)
        if path.suffix.lower() == ".zip":
            from raglet.storage.zip_backend import ZipStorageBackend

            return ZipStorageBackend()

        # Directory format (if exists and is directory)
        if path.exists() and path.is_dir():
            from raglet.storage.directory_backend import DirectoryStorageBackend

            return DirectoryStorageBackend()

        # Check if existing file is SQLite or Zip (magic bytes)
        if path.exists() and path.is_file():
            try:
                with open(path, "rb") as f:
                    header = f.read(16)
                    if header.startswith(b"SQLite format"):
                        from raglet.storage.sqlite_backend import SQLiteStorageBackend

                        return SQLiteStorageBackend()
                    elif header.startswith(b"PK"):  # Zip magic bytes
                        from raglet.storage.zip_backend import ZipStorageBackend

                        return ZipStorageBackend()
            except Exception:
                pass

        # Default: Directory format (will create directory if needed)
        from raglet.storage.directory_backend import DirectoryStorageBackend

        return DirectoryStorageBackend()

    def _get_default_backend(self, file_path: str) -> "StorageBackend":
        """Get default storage backend.

        Args:
            file_path: Path to file

        Returns:
            StorageBackend instance
        """
        return RAGlet._detect_backend(file_path)
