"""Data models for RAG embedding representations.

Defines the core data structures produced by embedding models and services,
preserving the provenance link to source Chunks without duplicating chunk content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EmbeddingResult:
    """Represents the vector embedding of a source Chunk.

    Preserves the provenance link to the source Chunk via chunk_id while
    keeping embedding representations decoupled from raw chunk text.
    """

    chunk_id: str
    vector: List[float]
    dimension: int
    model_name: str
    is_normalized: bool = True
    token_count: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate embedding result attributes upon construction."""
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")

        if not isinstance(self.vector, (list, tuple)):
            raise TypeError(f"vector must be a sequence of floats, got {type(self.vector).__name__}")

        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {self.dimension}")

        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Vector length ({len(self.vector)}) does not match declared dimension ({self.dimension})"
            )

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if not isinstance(self.is_normalized, bool):
            raise TypeError(f"is_normalized must be a boolean, got {type(self.is_normalized).__name__}")

        if self.token_count is not None:
            if not isinstance(self.token_count, int) or self.token_count < 0:
                raise ValueError(f"token_count must be a non-negative integer or None, got {self.token_count}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert EmbeddingResult to a dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "vector": list(self.vector),
            "dimension": self.dimension,
            "model_name": self.model_name,
            "is_normalized": self.is_normalized,
            "token_count": self.token_count,
        }
