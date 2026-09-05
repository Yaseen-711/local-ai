"""PostgreSQL + pgvector implementation of the domain VectorStore protocol."""

from __future__ import annotations

from typing import List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from rag.domain.interfaces import VectorStore
from rag.domain.models import Chunk, Document, RetrievalResult
from rag.storage.database import DatabaseManager
from rag.storage.models import (
    ChunkModel,
    DocumentModel,
    EMBEDDING_DIMENSION,
)


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector backed vector store satisfying domain VectorStore protocol.

    Translates domain objects (Document, Chunk) to SQLAlchemy ORM models, persists them
    with 768-dimensional embeddings, and executes vector similarity searches using
    pgvector's cosine distance operator.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize the vector store with a DatabaseManager instance."""
        self._db = db_manager

    @property
    def db(self) -> DatabaseManager:
        """Access the underlying DatabaseManager."""
        return self._db

    def add_document(self, document: Document) -> None:
        """Store or update a domain Document in the database."""
        with self._db.session() as session:
            stmt = (
                insert(DocumentModel)
                .values(
                    id=document.id,
                    content=document.content,
                    metadata_=document.metadata,
                )
                .on_conflict_do_update(
                    index_elements=[DocumentModel.id],
                    set_={
                        "content": document.content,
                        "metadata": document.metadata,
                    },
                )
            )
            session.execute(stmt)

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[List[float]],
    ) -> None:
        """Store chunks alongside their corresponding vector embeddings.

        Args:
            chunks: Sequence of domain Chunk objects.
            embeddings: Corresponding 768-dimensional vector embeddings.

        Raises:
            ValueError: If lengths of chunks and embeddings do not match, or if any
                embedding does not match the expected dimension (768).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between number of chunks ({len(chunks)}) and "
                f"embeddings ({len(embeddings)})."
            )

        for idx, emb in enumerate(embeddings):
            if len(emb) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding at index {idx} has invalid dimension {len(emb)}. "
                    f"Expected {EMBEDDING_DIMENSION} dimensions."
                )

        if not chunks:
            return

        with self._db.session() as session:
            # Ensure parent documents exist to satisfy foreign key constraints
            doc_ids = {chunk.document_id for chunk in chunks}
            existing_docs = set(
                session.scalars(
                    select(DocumentModel.id).where(DocumentModel.id.in_(doc_ids))
                ).all()
            )
            missing_doc_ids = doc_ids - existing_docs
            for missing_id in missing_doc_ids:
                session.add(
                    DocumentModel(
                        id=missing_id,
                        content="",
                        metadata_={"auto_generated": True},
                    )
                )
            session.flush()

            # Insert or update chunks
            for chunk, embedding in zip(chunks, embeddings):
                stmt = (
                    insert(ChunkModel)
                    .values(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata_=chunk.metadata,
                        chunk_index=chunk.metadata.get("chunk_index", 0),
                        embedding=embedding,
                    )
                    .on_conflict_do_update(
                        index_elements=[ChunkModel.id],
                        set_={
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "chunk_index": chunk.metadata.get("chunk_index", 0),
                            "embedding": embedding,
                        },
                    )
                )
                session.execute(stmt)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Execute vector similarity search using cosine distance and return top-k matches.

        Args:
            query_embedding: 768-dimensional query vector.
            top_k: Maximum number of relevant chunks to retrieve (default: 5).

        Returns:
            List of domain RetrievalResult objects sorted by relevance score descending.

        Raises:
            ValueError: If query_embedding does not match the expected dimension (768),
                or if top_k is less than 1.
        """
        if len(query_embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Query embedding has invalid dimension {len(query_embedding)}. "
                f"Expected {EMBEDDING_DIMENSION} dimensions."
            )

        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        with self._db.session() as session:
            # Cosine distance operator in pgvector (<=>)
            distance_expr = ChunkModel.embedding.cosine_distance(query_embedding).label("distance")
            stmt = (
                select(ChunkModel, distance_expr)
                .order_by(distance_expr.asc())
                .limit(top_k)
            )

            rows = session.execute(stmt).all()

            results: List[RetrievalResult] = []
            for chunk_row, distance in rows:
                domain_chunk = Chunk(
                    id=chunk_row.id,
                    document_id=chunk_row.document_id,
                    content=chunk_row.content,
                    metadata=dict(chunk_row.metadata_),
                )
                # Cosine distance = 1 - cosine_similarity; convert to similarity score in [0.0, 1.0]
                similarity_score = max(0.0, 1.0 - float(distance))
                results.append(
                    RetrievalResult(
                        chunk=domain_chunk,
                        score=similarity_score,
                    )
                )

            return results
