"""High-level orchestration service for candidate reranking.

Coordinates optional second-stage cross-encoder reranking, allowing Adaptive RAG
pipelines to bypass or apply reranking dynamically without breaking result schemas.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.reranking.interfaces import Reranker
from rag.reranking.models import RankedChunk
from rag.retrieval.models import RetrievedChunk


class RerankingService:
    """Orchestration service for second-stage candidate reranking."""

    def __init__(self, reranker: Optional[Reranker] = None) -> None:
        """Initialize RerankingService.

        Args:
            reranker: Concrete Reranker implementation. Defaults to CrossEncoderReranker.
        """
        self.reranker = reranker or CrossEncoderReranker()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RankedChunk]:
        """Execute reranking on candidates using the configured model.

        Args:
            query: User query text.
            candidates: Sequence of RetrievedChunk instances.
            top_n: Optional count to truncate final results to.

        Returns:
            List of RankedChunk instances.
        """
        return self.reranker.rerank(query=query, candidates=candidates, top_n=top_n)

    def rerank_if_enabled(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        enabled: bool = True,
        top_n: Optional[int] = None,
    ) -> List[RankedChunk]:
        """Conditionally apply reranking or pass through original retrieval order.

        Args:
            query: User query text.
            candidates: Sequence of RetrievedChunk instances.
            enabled: If True, executes cross-encoder reranking. If False, preserves
                the original vector retrieval rank and similarity score.
            top_n: Optional truncation count.

        Returns:
            List of RankedChunk instances.
        """
        if not candidates:
            return []

        if enabled:
            return self.rerank(query=query, candidates=candidates, top_n=top_n)

        # Pass-through: convert RetrievedChunk to RankedChunk preserving original metrics
        selected = candidates[:top_n] if top_n is not None else candidates
        return [
            RankedChunk.from_retrieved_chunk(
                candidate=c,
                reranking_score=c.similarity_score,
                rerank_rank=c.rank,
            )
            for c in selected
        ]
