"""Framework-independent protocols and interfaces for RAG capabilities.

These structural protocols define WHAT RAG components do (embedding generation,
vector storage and retrieval) without specifying HOW they are implemented.
"""

from typing import List, Protocol, Sequence, runtime_checkable

from rag.domain.models import Chunk, RetrievalResult


@runtime_checkable
class Embedder(Protocol):
    """Protocol for components capable of converting text into vector embeddings."""

    def embed_text(self, text: str) -> List[float]:
        """Convert a single text string into a vector embedding float array."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Convert a sequence of text strings into vector embeddings."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector databases capable of storing and searching chunk embeddings."""

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[List[float]],
    ) -> None:
        """Store chunks alongside their corresponding vector embeddings."""
        ...

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Execute vector similarity search using a query vector and return top-k matches."""
        ...
