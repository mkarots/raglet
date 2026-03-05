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

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "size": self.size,
            "overlap": self.overlap,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkingConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            ChunkingConfig instance
        """
        return cls(
            size=data.get("size", 512),
            overlap=data.get("overlap", 50),
            strategy=data.get("strategy", "sentence-aware"),
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"
    normalize: bool = True  # Default to True for cosine similarity

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

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "model": self.model,
            "batch_size": self.batch_size,
            "device": self.device,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            EmbeddingConfig instance
        """
        return cls(
            model=data.get("model", "all-MiniLM-L6-v2"),
            batch_size=data.get("batch_size", 32),
            device=data.get("device", "cpu"),
            normalize=data.get("normalize", True),  # Default to True for cosine similarity
        )


@dataclass
class SearchConfig:
    """Configuration for vector search."""

    default_top_k: int = 5
    similarity_threshold: Optional[float] = None
    index_type: str = "flat_ip"  # Cosine similarity (IndexFlatIP)

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
        if self.index_type != "flat_ip":
            raise ValueError(
                f"Invalid index_type: {self.index_type}. Only 'flat_ip' (cosine similarity) is supported."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        result = {
            "default_top_k": self.default_top_k,
            "index_type": self.index_type,
        }
        if self.similarity_threshold is not None:
            result["similarity_threshold"] = self.similarity_threshold
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            SearchConfig instance
        """
        return cls(
            default_top_k=data.get("default_top_k", 5),
            similarity_threshold=data.get("similarity_threshold"),
            index_type=data.get("index_type", "flat_ip"),  # Default to cosine similarity
        )


@dataclass
class RAGletConfig:
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

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config with nested configs
        """
        return {
            "chunking": self.chunking.to_dict(),
            "embedding": self.embedding.to_dict(),
            "search": self.search.to_dict(),
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGletConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values (may include nested configs)

        Returns:
            RAGletConfig instance
        """
        chunking_data = data.get("chunking", {})
        embedding_data = data.get("embedding", {})
        search_data = data.get("search", {})

        return cls(
            chunking=ChunkingConfig.from_dict(chunking_data),
            embedding=EmbeddingConfig.from_dict(embedding_data),
            search=SearchConfig.from_dict(search_data),
            custom_metadata=data.get("custom_metadata", {}),
        )
