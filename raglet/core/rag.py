"""TinyRAG main class."""

from typing import Optional

from raglet.config.config import TinyRAGConfig
from raglet.core.chunk import Chunk
from raglet.embeddings.interfaces import EmbeddingGenerator
from raglet.processing.interfaces import Chunker, DocumentExtractor
from raglet.vector_store.interfaces import VectorStore


class TinyRAG:
    """Main RAG orchestrator class."""

    def __init__(
        self,
        chunks: list[Chunk],
        config: TinyRAGConfig,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """Initialize TinyRAG.

        Args:
            chunks: List of chunks
            config: Configuration
            embedding_generator: Optional embedding generator (created if None)
            vector_store: Optional vector store (created if None)
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

        # Generate embeddings and add to vector store if chunks exist
        if chunks:
            embeddings = self.embedding_generator.generate(chunks)
            self.vector_store.add_vectors(embeddings, chunks)

    @classmethod
    def from_files(
        cls,
        files: list[str],
        document_extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[Chunker] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[TinyRAGConfig] = None,
    ) -> "TinyRAG":
        """Create TinyRAG from files.

        Full pipeline: Extract → Chunk → Embed → Index

        Args:
            files: List of file paths
            document_extractor: Optional document extractor (uses factory if None)
            chunker: Optional chunker (creates default if None)
            embedding_generator: Optional embedding generator (creates default if None)
            vector_store: Optional vector store (creates default if None)
            config: Optional configuration (uses defaults if None)

        Returns:
            TinyRAG instance with searchable index

        Raises:
            FileNotFoundError: If any file doesn't exist
            ValueError: If configuration is invalid or extraction fails
        """
        if config is None:
            config = TinyRAGConfig()

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

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Apply similarity threshold if configured
        # Note: FAISS returns L2 distances (lower = more similar)
        # We store negative distance as score (so higher score = more similar)
        # Threshold is typically a minimum similarity (0-1), so we need to convert
        # For now, if threshold is set, we'll filter by ensuring score is above threshold
        # (This is a simplified approach - proper similarity conversion would need
        #  distance-to-similarity mapping based on the distance distribution)
        if threshold is not None:
            # Filter results where score meets threshold
            # Since scores are negative distances, we compare against negative threshold
            results = [r for r in results if r.score is not None and r.score >= -threshold]

        return results

    def get_all_chunks(self) -> list[Chunk]:
        """Get all chunks.

        Returns:
            List of all chunks
        """
        return self.chunks
