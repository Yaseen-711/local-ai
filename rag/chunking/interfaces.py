"""Protocol interfaces for RAG document chunking strategies.

Defines the contract for converting a NormalizedDocument into a sequence of
rag.domain.models.Chunk objects ready for downstream metadata propagation,
embeddings, and vector storage.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from rag.domain.models import Chunk
from rag.normalization.models import NormalizedDocument


@runtime_checkable
class DocumentChunker(Protocol):
    """Protocol for strategies that partition a NormalizedDocument into Chunks."""

    def chunk(self, document: NormalizedDocument) -> List[Chunk]:
        """Convert a normalized document into a list of structural domain Chunks.

        Args:
            document: Clean, normalized document containing ordered structural elements.

        Returns:
            List of domain Chunk objects with deterministic IDs, content, and metadata.
        """
        ...
