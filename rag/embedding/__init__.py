"""Embedding layer for Local AI Foundation RAG subsystem."""

from rag.embedding.interfaces import EmbeddingModel
from rag.embedding.models import EmbeddingResult
from rag.embedding.nomic import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    NomicEmbedder,
    NomicEmbeddingModel,
    QUERY_PREFIX,
)
from rag.embedding.service import ChunkEmbeddingService

__all__ = [
    "DOCUMENT_PREFIX",
    "EMBEDDING_DIMENSION",
    "ChunkEmbeddingService",
    "EmbeddingModel",
    "EmbeddingResult",
    "NomicEmbedder",
    "NomicEmbeddingModel",
    "QUERY_PREFIX",
]
