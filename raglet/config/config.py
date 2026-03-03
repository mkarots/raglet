"""Configuration classes."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    size: int = 512
    overlap: int = 50
    strategy: str = "sentence-aware"

    def validate(self) -> None:
        """Validate chunking configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.overlap >= self.size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.strategy not in ["fixed", "sentence-aware", "semantic"]:
            raise ValueError(f"Invalid chunk_strategy: {self.strategy}")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"
    normalize: bool = False

    def validate(self) -> None:
        """Validate embedding configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.model:
            raise ValueError("embedding model must be specified")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.device not in ["cpu", "cuda"]:
            raise ValueError("device must be 'cpu' or 'cuda'")


@dataclass
class SearchConfig:
    """Configuration for vector search."""

    default_top_k: int = 5
    similarity_threshold: Optional[float] = None
    index_type: str = "flat_l2"

    def validate(self) -> None:
        """Validate search configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.default_top_k < 1:
            raise ValueError("default_top_k must be >= 1")
        if self.similarity_threshold is not None:
            if not 0.0 <= self.similarity_threshold <= 1.0:
                raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.index_type not in ["flat_l2"]:
            raise ValueError(f"Invalid index_type: {self.index_type}")


@dataclass
class TinyRAGConfig:
    """Main configuration class."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate entire configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        self.chunking.validate()
        self.embedding.validate()
        self.search.validate()
