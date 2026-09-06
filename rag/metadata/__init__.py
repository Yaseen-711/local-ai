"""RAG metadata model and propagation pipeline.

Provides structured, typed metadata contracts across document, element, and chunk
layers, with first-class provenance tracking and backwards-compatible dictionary
serialization.
"""

from rag.metadata.models import (
    ChunkMetadata,
    DocumentMetadata,
    ElementMetadata,
    ProvenanceMetadata,
)
from rag.metadata.pipeline import MetadataPipeline

__all__ = [
    "ChunkMetadata",
    "DocumentMetadata",
    "ElementMetadata",
    "MetadataPipeline",
    "ProvenanceMetadata",
]
