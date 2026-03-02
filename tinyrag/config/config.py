"""Configuration classes."""

from dataclasses import dataclass, field
from typing import Any


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
class TinyRAGConfig:
    """Main configuration class."""

    """Main configuration class."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate entire configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        self.chunking.validate()
