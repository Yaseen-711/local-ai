"""RAG document chunking subsystem.

Provides structural document chunking that converts NormalizedDocument objects
into discrete, semantically coherent rag.domain.models.Chunk objects ready for
embedding and indexing.
"""

from rag.chunking.interfaces import DocumentChunker
from rag.chunking.options import ChunkingOptions, FallbackSplitStrategy
from rag.chunking.structural import StructuralChunker

__all__ = [
    "ChunkingOptions",
    "DocumentChunker",
    "FallbackSplitStrategy",
    "StructuralChunker",
]
