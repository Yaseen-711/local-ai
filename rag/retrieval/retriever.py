"""PostgreSQL + pgvector implementation of the VectorRetriever protocol.

Provides PgVectorRetriever to perform semantic similarity search over persisted
ChunkModel embeddings using pgvector's cosine distance operator.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select

from rag.retrieval.interfaces import VectorRetriever
from rag.retrieval.models import RetrievedChunk
from rag.storage.database import DatabaseManager
from rag.storage.models import ChunkModel, EMBEDDING_DIMENSION


class PgVectorRetriever(VectorRetriever):
    """Semantic vector retriever backed by PostgreSQL and pgvector.

    Features:
    - Pure read-only vector similarity search.
    - Cosine similarity scoring (s = 1 - distance).
    - Database-level Top-K truncation and stable tie-breaking.
    - Optional similarity threshold filtering.
    - Optional document scoping and JSONB metadata filtering.
    - Zero coupling to specific embedding models.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        """Initialize PgVectorRetriever.

        Args:
            db_manager: DatabaseManager managing PostgreSQL engine and sessions.
            dimension: Expected query vector dimension (default: 768).
        """
        self._db = db_manager
        self._dimension = dimension

    @property
    def db(self) -> DatabaseManager:
        """Access the underlying DatabaseManager."""
        return self._db

    @property
    def dimension(self) -> int:
        """Expected vector embedding dimension."""
        return self._dimension

    def _validate_inputs(
        self,
        query_vector: Sequence[float],
        top_k: int,
        document_id: Optional[str],
        similarity_threshold: Optional[float],
        filters: Optional[Dict[str, Any]],
    ) -> List[float]:
        """Validate query parameters and return sanitized float vector."""
        if not isinstance(query_vector, Sequence):
            raise TypeError(f"query_vector must be a Sequence, got {type(query_vector).__name__}")

        if len(query_vector) == 0:
            raise ValueError("query_vector must not be empty")

        if len(query_vector) != self._dimension:
            raise ValueError(
                f"Query vector dimension ({len(query_vector)}) does not match "
                f"retriever dimension ({self._dimension})"
            )

        sanitized_vector: List[float] = []
        for idx, val in enumerate(query_vector):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(
                    f"Query vector contains non-finite value '{val}' at position {idx}"
                )
            sanitized_vector.append(float(val))

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if similarity_threshold is not None:
            if not isinstance(similarity_threshold, (int, float)):
                raise TypeError(
                    f"similarity_threshold must be a float, got {type(similarity_threshold).__name__}"
                )
            if not -1.0 <= similarity_threshold <= 1.0:
                raise ValueError(
                    f"similarity_threshold must be in range [-1.0, 1.0], got {similarity_threshold}"
                )

        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("document_id must be a non-empty string if provided")

        if filters is not None and not isinstance(filters, dict):
            raise TypeError(f"filters must be a dict, got {type(filters).__name__}")

        return sanitized_vector

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
        vec = self._validate_inputs(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            similarity_threshold=similarity_threshold,
            filters=filters,
        )

        with self._db.session() as session:
            # Cosine distance operator in pgvector (<=>)
            distance_expr = ChunkModel.embedding.cosine_distance(vec).label("distance")
            stmt = select(ChunkModel, distance_expr)

            # Document scoping
            if document_id is not None:
                stmt = stmt.where(ChunkModel.document_id == document_id.strip())

            # Metadata containment filtering (JSONB @> operator)
            if filters:
                stmt = stmt.where(ChunkModel.metadata_.contains(filters))

            # Similarity threshold filtering
            # In pgvector: distance = 1 - similarity  =>  similarity >= threshold <=> distance <= 1 - threshold
            if similarity_threshold is not None:
                max_distance = 1.0 - float(similarity_threshold)
                stmt = stmt.where(distance_expr <= max_distance)

            # Stable deterministic ordering: primary by distance ASC, secondary by chunk ID ASC
            stmt = stmt.order_by(distance_expr.asc(), ChunkModel.id.asc()).limit(top_k)

            rows = session.execute(stmt).all()
            if not rows:
                return []

            results: List[RetrievedChunk] = []
            for rank_idx, (chunk_row, dist) in enumerate(rows, start=1):
                # Convert cosine distance to cosine similarity: s = 1 - d
                similarity_score = 1.0 - float(dist)

                results.append(
                    RetrievedChunk(
                        chunk_id=chunk_row.id,
                        document_id=chunk_row.document_id,
                        content=chunk_row.content,
                        metadata=dict(chunk_row.metadata_),
                        similarity_score=similarity_score,
                        rank=rank_idx,
                        chunk_index=chunk_row.chunk_index,
                    )
                )

            return results
