"""Domain and result data structures for RAG vector retrieval.

Provides typed, immutable representations of chunks retrieved via vector
similarity search, maintaining provenance and search relevance rankings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from rag.domain.models import Chunk, RetrievalResult


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a chunk retrieved through semantic similarity search.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        document_id: Identifier of the parent source document.
        content: Unmodified textual content of the chunk.
        metadata: Preserved document, structural, and provenance metadata.
        similarity_score: Cosine similarity score in range [-1.0, 1.0].
        rank: 1-indexed relevance rank in retrieval results (1 = most relevant).
        chunk_index: Sequential index of the chunk within the document.
    """

    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0
    rank: int = 1
    chunk_index: int = 0

    def __post_init__(self) -> None:
        """Validate attribute constraints upon creation."""
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")

        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")

        if not isinstance(self.content, str):
            raise TypeError(f"content must be a string, got {type(self.content).__name__}")

        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(self.metadata).__name__}")

        if not isinstance(self.similarity_score, (int, float)):
            raise TypeError(f"similarity_score must be a float, got {type(self.similarity_score).__name__}")

        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be a positive integer (>= 1), got {self.rank}")

    def to_domain_chunk(self) -> Chunk:
        """Convert to pure domain Chunk model."""
        return Chunk(
            id=self.chunk_id,
            document_id=self.document_id,
            content=self.content,
            metadata=dict(self.metadata),
        )

    def to_retrieval_result(self) -> RetrievalResult:
        """Convert to pure domain RetrievalResult model."""
        return RetrievalResult(
            chunk=self.to_domain_chunk(),
            score=self.similarity_score,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RetrievedChunk to a dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "chunk_index": self.chunk_index,
        }
