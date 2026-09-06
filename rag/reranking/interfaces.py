"""Protocol interfaces for RAG candidate reranking.

Defines the contract for scoring and re-ordering retrieved candidate chunks
using query-document cross-attention models.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, runtime_checkable

from rag.reranking.models import RankedChunk
from rag.retrieval.models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Protocol for candidate rerankers.

    Consumes candidate chunks from first-stage retrieval and produces re-ordered
    RankedChunk results based on deep query-document relevance scoring.
    """

    @property
    def model_name(self) -> str:
        """Name or identifier of the underlying reranker model."""
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RankedChunk]:
        """Score and re-order candidate chunks relative to a query.

        Args:
            query: Raw query text.
            candidates: Sequence of RetrievedChunk candidates to rerank.
            top_n: Optional maximum number of top candidates to return.

        Returns:
            List of RankedChunk objects ordered by reranking_score descending.
        """
        ...
