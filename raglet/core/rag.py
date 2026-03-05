"""RAGlet main class."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from raglet.config.config import RAGletConfig
from raglet.core.chunk import Chunk
from raglet.embeddings.interfaces import EmbeddingGenerator
from raglet.processing.interfaces import Chunker, DocumentExtractor
from raglet.vector_store.interfaces import VectorStore

if TYPE_CHECKING:
    from raglet.storage.interfaces import StorageBackend


class RAGlet:
    """Main RAG orchestrator class."""

    def __init__(
        self,
        chunks: list[Chunk],
        config: RAGletConfig,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        embeddings: Optional[np.ndarray] = None,
    ):
        """Initialize RAGlet.

        Args:
            chunks: List of chunks
            config: Configuration
            embedding_generator: Optional embedding generator (created if None)
            vector_store: Optional vector store (created if None)
            embeddings: Optional pre-computed embeddings (generated if None and chunks exist)
        """
        self.chunks = chunks
        self.config = config

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
    ) -> "RAGlet":
        """Create RAGlet from files.

        Full pipeline: Extract → Chunk → Embed → Index

        Args:
            files: List of file paths
            document_extractor: Optional document extractor (uses factory if None)
            chunker: Optional chunker (creates default if None)
            embedding_generator: Optional embedding generator (creates default if None)
            vector_store: Optional vector store (creates default if None)
            config: Optional configuration (uses defaults if None)

        Returns:
            RAGlet instance with searchable index

        Raises:
            FileNotFoundError: If any file doesn't exist
            ValueError: If configuration is invalid or extraction fails
        """
        if config is None:
            config = RAGletConfig()

        config.validate()

        # Step 1: Extract text from files
        from raglet.processing.chunker import SentenceAwareChunker
        from raglet.processing.extractor_factory import create_extractor

        all_chunks = []

        for file_path in files:
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

        # Step 3 & 4: Generate embeddings and create vector store
        # (handled in __init__)
        return cls(
            chunks=all_chunks,
            config=config,
            embedding_generator=embedding_generator,
            vector_store=vector_store,
        )

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
    ) -> None:
        """Add files to existing RAGlet.

        Extracts, chunks, and embeds new files, then adds them to the existing RAGlet.

        Args:
            files: List of file paths to add
            document_extractor: Optional document extractor (uses factory if None)
            chunker: Optional chunker (uses RAGlet's config if None)
            file_path: Optional file path (if provided, saves immediately)

        Raises:
            FileNotFoundError: If any file doesn't exist
            ValueError: If extraction fails
        """
        if not files:
            return

        # Extract and chunk files (similar to from_files logic)
        from raglet.processing.chunker import SentenceAwareChunker
        from raglet.processing.extractor_factory import create_extractor

        new_chunks = []

        for file_path_item in files:
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

        # Add chunks (reuse add_chunks logic)
        if new_chunks:
            self.add_chunks(new_chunks, file_path=file_path)

    def add_chunks(
        self,
        new_chunks: list[Chunk],
        file_path: Optional[str] = None,
    ) -> None:
        """Add chunks incrementally.

        Args:
            new_chunks: New chunks to add
            file_path: Optional file path (if provided, saves immediately)
        """
        if not new_chunks:
            return

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
