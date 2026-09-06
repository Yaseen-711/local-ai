"""Protocol interfaces for RAG embedding models.

Defines the contract for models capable of converting text sequences into dense vector
representations for document chunks and queries.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for models capable of generating dense vector embeddings from text."""

    @property
    def dimension(self) -> int:
        """The dimensionality of the output embeddings (e.g. 768)."""
        ...

    @property
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        ...

    @property
    def is_normalized(self) -> bool:
        """Whether the produced vector embeddings are L2 unit-normalized."""
        ...

    def embed_documents(self, document_texts: Sequence[str]) -> List[List[float]]:
        """Convert a sequence of document texts into vector embeddings.

        Applies appropriate document-specific prefix/formatting if required by the model.

        Args:
            document_texts: Sequence of text contents to embed.

        Returns:
            List of float vector embeddings, matching input length and order.
        """
        ...

    def embed_query(self, query_text: str) -> List[float]:
        """Convert a single query text into a vector embedding.

        Applies appropriate query-specific prefix/formatting if required by the model.

        Args:
            query_text: Query string to embed.

        Returns:
            Float vector embedding of length `dimension`.
        """
        ...
