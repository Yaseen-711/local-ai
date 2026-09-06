"""Interfaces and protocols for document normalization."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.ingestion.models import IngestedDocument
from rag.normalization.models import NormalizedDocument


@runtime_checkable
class DocumentNormalizer(Protocol):
    """Protocol for components that normalize raw ingested documents."""

    def normalize(self, document: IngestedDocument) -> NormalizedDocument:
        """Normalize an ingested document into a clean, Docling-independent representation.

        Args:
            document: Raw IngestedDocument from the ingestion stage.

        Returns:
            NormalizedDocument with cleaned whitespace, validated elements, and preserved structure.
        """
        ...
