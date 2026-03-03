"""Chunk domain model."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "index": self.index,
            "metadata": self.metadata,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            text=data["text"],
            source=data["source"],
            index=data["index"],
            metadata=data.get("metadata", {}),
            score=data.get("score"),
        )
