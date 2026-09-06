"""PostgreSQL + pgvector implementation of the VectorIndexer protocol.

Provides PgVectorIndexer to persist domain Chunks alongside their computed
EmbeddingResults using atomic transactions, idempotent upserts, and strict validation.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from rag.domain.models import Chunk, Document
from rag.embedding.models import EmbeddingResult
from rag.indexing.interfaces import VectorIndexer
from rag.storage.database import DatabaseManager
from rag.storage.models import (
    ChunkModel,
    DocumentModel,
    EMBEDDING_DIMENSION,
)


class PgVectorIndexer(VectorIndexer):
    """Vector indexer implementation backed by PostgreSQL and pgvector.

    Features:
    - Atomic transactional batch persistence.
    - Idempotent upserts (ON CONFLICT DO UPDATE).
    - Explicit Chunk ↔ EmbeddingResult identity validation.
    - Strict vector validation: finite floats, non-empty, dimension matching.
    - Full metadata and provenance preservation.
    - Cascade and granular deletion capabilities.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        """Initialize PgVectorIndexer.

        Args:
            db_manager: DatabaseManager managing PostgreSQL engine and sessions.
            dimension: Expected vector dimension (defaults to 768).
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

    def _validate_pair(
        self,
        chunk: Chunk,
        embedding: EmbeddingResult,
        index: int,
    ) -> Dict[str, Any]:
        """Validate a Chunk and its corresponding EmbeddingResult.

        Returns:
            Prepared metadata dictionary enriched with model information.
        """
        if not isinstance(chunk, Chunk):
            raise TypeError(f"Item at index {index} is not a Chunk instance: {type(chunk).__name__}")

        if not isinstance(embedding, EmbeddingResult):
            raise TypeError(
                f"Embedding at index {index} is not an EmbeddingResult instance: {type(embedding).__name__}"
            )

        if not chunk.id or not str(chunk.id).strip():
            raise ValueError(f"Chunk at index {index} has empty id")

        if not chunk.document_id or not str(chunk.document_id).strip():
            raise ValueError(f"Chunk '{chunk.id}' at index {index} has empty document_id")

        if not chunk.content or not chunk.content.strip():
            raise ValueError(f"Chunk '{chunk.id}' at index {index} has empty or whitespace-only content")

        # Explicit identity match: Never rely on sequence order alone
        if chunk.id != embedding.chunk_id:
            raise ValueError(
                f"Identity mismatch at index {index}: Chunk ID '{chunk.id}' != "
                f"EmbeddingResult chunk_id '{embedding.chunk_id}'"
            )

        # Dimension validation
        if embedding.dimension != self._dimension:
            raise ValueError(
                f"Embedding dimension {embedding.dimension} for chunk '{chunk.id}' "
                f"does not match indexer dimension {self._dimension}"
            )

        if len(embedding.vector) != self._dimension:
            raise ValueError(
                f"Vector length {len(embedding.vector)} for chunk '{chunk.id}' "
                f"does not match indexer dimension {self._dimension}"
            )

        # Non-finite value check (reject NaN, Inf, -Inf)
        for i, val in enumerate(embedding.vector):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' contains invalid non-finite value '{val}' at position {i}"
                )

        # Metadata serializability validation
        if not isinstance(chunk.metadata, dict):
            raise TypeError(f"Chunk '{chunk.id}' metadata must be a dict, got {type(chunk.metadata).__name__}")

        try:
            # Test JSON serializability
            json.dumps(chunk.metadata)
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"Chunk '{chunk.id}' metadata is not JSON serializable") from exc

        # Prepare metadata: preserve all chunk provenance while tagging model info
        meta = dict(chunk.metadata)
        meta.setdefault("embedding_model", embedding.model_name)
        meta.setdefault("is_normalized", embedding.is_normalized)
        if embedding.token_count is not None:
            meta.setdefault("token_count", embedding.token_count)

        return meta

    def index_chunk(self, chunk: Chunk, embedding: EmbeddingResult) -> None:
        """Persist or update a single Chunk with its EmbeddingResult.

        Args:
            chunk: Source domain Chunk.
            embedding: Corresponding EmbeddingResult.
        """
        self.index_chunks([chunk], [embedding])

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> int:
        """Persist or update a batch of Chunks with their EmbeddingResults.

        Operates inside a single database transaction. Idempotently upserts
        chunks on primary key conflict.

        Args:
            chunks: Sequence of domain Chunks.
            embeddings: Sequence of EmbeddingResults.

        Returns:
            Number of chunks successfully indexed.
        """
        if not isinstance(chunks, Sequence):
            raise TypeError(f"chunks must be a Sequence, got {type(chunks).__name__}")
        if not isinstance(embeddings, Sequence):
            raise TypeError(f"embeddings must be a Sequence, got {type(embeddings).__name__}")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            )

        if not chunks:
            return 0

        # Validate all pairs before touching the database
        prepared_data: List[tuple[Chunk, EmbeddingResult, Dict[str, Any]]] = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            enriched_meta = self._validate_pair(chunk, embedding, idx)
            prepared_data.append((chunk, embedding, enriched_meta))

        with self._db.session() as session:
            # 1. Ensure all referenced parent documents exist to satisfy FK constraints
            doc_ids = {c.document_id for c, _, _ in prepared_data}
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
            if missing_doc_ids:
                session.flush()

            # 2. Idempotent upsert of all chunks
            for chunk, embedding, meta in prepared_data:
                chunk_index = int(meta.get("chunk_index", 0))
                stmt = (
                    insert(ChunkModel)
                    .values(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata_=meta,
                        chunk_index=chunk_index,
                        embedding=list(embedding.vector),
                    )
                    .on_conflict_do_update(
                        index_elements=[ChunkModel.id],
                        set_={
                            "document_id": chunk.document_id,
                            "content": chunk.content,
                            "metadata": meta,
                            "chunk_index": chunk_index,
                            "embedding": list(embedding.vector),
                        },
                    )
                )
                session.execute(stmt)

        return len(prepared_data)

    def index_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> int:
        """Persist or update a parent Document and all its embedded Chunks.

        Args:
            document: Parent domain Document.
            chunks: Sequence of domain Chunk objects belonging to the document.
            embeddings: Sequence of EmbeddingResult objects for the chunks.

        Returns:
            Number of chunks successfully indexed.
        """
        if not isinstance(document, Document):
            raise TypeError(f"document must be a Document instance, got {type(document).__name__}")
        if not document.id or not str(document.id).strip():
            raise ValueError("Document id must be a non-empty string")

        if not isinstance(document.metadata, dict):
            raise TypeError("Document metadata must be a dict")

        # Validate chunk parent document association
        for i, c in enumerate(chunks):
            if c.document_id != document.id:
                raise ValueError(
                    f"Chunk '{c.id}' at index {i} belongs to document '{c.document_id}', "
                    f"expected '{document.id}'"
                )

        # Upsert document first, then index chunks in the same transaction
        with self._db.session() as session:
            doc_stmt = (
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
            session.execute(doc_stmt)

        # Index chunks with parent guaranteed to exist
        return self.index_chunks(chunks, embeddings)

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk by its identifier.

        Args:
            chunk_id: Identifier of the chunk to delete.

        Returns:
            True if the chunk was found and deleted, False otherwise.
        """
        if not chunk_id or not str(chunk_id).strip():
            raise ValueError("chunk_id must be a non-empty string")

        with self._db.session() as session:
            res = session.execute(
                delete(ChunkModel).where(ChunkModel.id == chunk_id)
            )
            return res.rowcount > 0

    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks associated with a document without deleting the document row.

        Args:
            document_id: Identifier of the parent document.

        Returns:
            Number of chunks deleted.
        """
        if not document_id or not str(document_id).strip():
            raise ValueError("document_id must be a non-empty string")

        with self._db.session() as session:
            res = session.execute(
                delete(ChunkModel).where(ChunkModel.document_id == document_id)
            )
            return res.rowcount

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its associated chunks (cascade delete).

        Args:
            document_id: Identifier of the document to delete.

        Returns:
            True if the document was found and deleted, False otherwise.
        """
        if not document_id or not str(document_id).strip():
            raise ValueError("document_id must be a non-empty string")

        with self._db.session() as session:
            res = session.execute(
                delete(DocumentModel).where(DocumentModel.id == document_id)
            )
            return res.rowcount > 0
