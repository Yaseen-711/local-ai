"""Data models and configuration for RAG candidate reranking.

Defines the typed, immutable RankedChunk representing a retrieved candidate after
cross-encoder relevance scoring and re-ordering, preserving original retrieval metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rag.retrieval.models import RetrievedChunk


@dataclass(frozen=True)
class RerankerConfig:
    """Configuration options for cross-encoder reranking models."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: Optional[str] = None
    batch_size: int = 32
    max_length: int = 512

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")


@dataclass(frozen=True)
class RankedChunk:
    """Represents a candidate chunk scored and ranked by a cross-encoder model.

    Maintains both original retrieval scores/ranks and the second-stage reranking
    scores/ranks for complete auditability and downstream processing.
    """

    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_similarity_score: float = 0.0
    original_retrieval_rank: int = 1
    reranking_score: float = 0.0
    rerank_rank: int = 1
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

        if not isinstance(self.original_similarity_score, (int, float)):
            raise TypeError(
                f"original_similarity_score must be a float, got {type(self.original_similarity_score).__name__}"
            )

        if not isinstance(self.original_retrieval_rank, int) or self.original_retrieval_rank < 1:
            raise ValueError(
                f"original_retrieval_rank must be a positive integer (>= 1), got {self.original_retrieval_rank}"
            )

        if not isinstance(self.reranking_score, (int, float)):
            raise TypeError(
                f"reranking_score must be a float, got {type(self.reranking_score).__name__}"
            )

        if not isinstance(self.rerank_rank, int) or self.rerank_rank < 1:
            raise ValueError(
                f"rerank_rank must be a positive integer (>= 1), got {self.rerank_rank}"
            )

    @classmethod
    def from_retrieved_chunk(
        cls,
        candidate: RetrievedChunk,
        reranking_score: float,
        rerank_rank: int,
    ) -> RankedChunk:
        """Construct RankedChunk from a RetrievedChunk and computed rerank score."""
        return cls(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            content=candidate.content,
            metadata=dict(candidate.metadata),
            original_similarity_score=candidate.similarity_score,
            original_retrieval_rank=candidate.rank,
            reranking_score=float(reranking_score),
            rerank_rank=rerank_rank,
            chunk_index=candidate.chunk_index,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RankedChunk to a dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "original_similarity_score": self.original_similarity_score,
            "original_retrieval_rank": self.original_retrieval_rank,
            "reranking_score": self.reranking_score,
            "rerank_rank": self.rerank_rank,
            "chunk_index": self.chunk_index,
        }
