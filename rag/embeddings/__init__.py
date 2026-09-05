"""Embedding subsystem for Local AI Foundation RAG."""

from rag.embeddings.nomic import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    NomicEmbedder,
    QUERY_PREFIX,
)

__all__ = [
    "DOCUMENT_PREFIX",
    "EMBEDDING_DIMENSION",
    "NomicEmbedder",
    "QUERY_PREFIX",
]
