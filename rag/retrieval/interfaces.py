"""Protocol interfaces for RAG vector retrieval.

Defines the contract for retrieving relevant Chunks using dense vector similarity
without coupling retrieval to specific embedding models, database drivers, or LLMs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from rag.retrieval.models import RetrievedChunk


@runtime_checkable
class VectorRetriever(Protocol):
    """Protocol for vector similarity search and retrieval backends.

    Responsible strictly for READ / SIMILARITY operations against persisted vector storage.
    Does not own embedding generation or reranking.
    """

    @property
    def dimension(self) -> int:
        """Expected dimensionality of query vectors."""
        ...

    def retrieve(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve top-K most similar chunks for a query vector.

        Args:
            query_vector: Dense float vector representing the query embedding.
            top_k: Maximum number of candidate chunks to return (default: 5).
            document_id: Optional document ID to restrict retrieval scope.
            similarity_threshold: Optional minimum cosine similarity score [-1.0, 1.0].
            filters: Optional dictionary of JSONB metadata key-value constraints.

        Returns:
            List of RetrievedChunk instances ordered by similarity descending.
        """
        ...
