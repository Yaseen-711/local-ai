"""Protocol interfaces for RAG vector indexing and persistence.

Defines the contract for persisting Chunk domain models and their corresponding
EmbeddingResults into vector storage without exposing retrieval/query capabilities.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from rag.domain.models import Chunk, Document
from rag.embedding.models import EmbeddingResult


@runtime_checkable
class VectorIndexer(Protocol):
    """Protocol for vector indexing and persistence backends.

    Responsible strictly for WRITE, UPDATE, and DELETE operations.
    READ and SIMILARITY operations are strictly decoupled and handled
    by downstream retrieval layers.
    """

    @property
    def dimension(self) -> int:
        """Expected vector embedding dimension."""
        ...

    def index_chunk(self, chunk: Chunk, embedding: EmbeddingResult) -> None:
        """Persist or update a single Chunk with its EmbeddingResult.

        Args:
            chunk: Source domain Chunk.
            embedding: Computed EmbeddingResult corresponding to the chunk.
        """
        ...

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> int:
        """Persist or update a batch of Chunks with their EmbeddingResults.

        Args:
            chunks: Sequence of domain Chunk objects.
            embeddings: Sequence of EmbeddingResult objects matching chunks by chunk_id.

        Returns:
            Number of chunks successfully indexed.
        """
        ...

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
        ...

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk by its identifier.

        Args:
            chunk_id: Identifier of the chunk to delete.

        Returns:
            True if the chunk was found and deleted, False otherwise.
        """
        ...

    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks associated with a document without deleting the document row.

        Args:
            document_id: Identifier of the parent document.

        Returns:
            Number of chunks deleted.
        """
        ...

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its associated chunks (cascade delete).

        Args:
            document_id: Identifier of the document to delete.

        Returns:
            True if the document was found and deleted, False otherwise.
        """
        ...
