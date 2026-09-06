"""Backwards-compatibility shim for rag.embeddings.nomic.

Re-exports from rag.embedding.nomic to preserve compatibility with existing
tests and components while transitioning to the unified rag.embedding package.
"""

from rag.embedding.nomic import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    NomicEmbedder,
    NomicEmbeddingModel,
    QUERY_PREFIX,
    l2_normalize,
)

__all__ = [
    "DOCUMENT_PREFIX",
    "EMBEDDING_DIMENSION",
    "NomicEmbedder",
    "NomicEmbeddingModel",
    "QUERY_PREFIX",
    "l2_normalize",
]
