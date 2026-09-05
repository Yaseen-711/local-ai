"""Normalized document representation for the RAG subsystem.

Provides clean, Docling-independent data structures representing a structured
document after text cleaning, whitespace normalization, and structural validation,
ready for consumption by the downstream chunking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.domain.models import Document


class NormalizedElementType(str, Enum):
    """Semantic type of a normalized document element."""

    TITLE = "title"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    OTHER = "other"


@dataclass(frozen=True)
class NormalizedElement:
    """A clean, normalized structural unit in the document.

    Attributes:
        index: 0-indexed sequential position of this element within the document.
        element_type: Semantic classification of the element.
        content: Cleaned and normalized text or markdown content.
        page_number: Optional 1-indexed page number if source preserved pages.
        heading_level: Heading level (1 for title, 1-6 for section headers).
        parent_heading: Content of the enclosing or most recent preceding heading.
        metadata: Domain and structural metadata (e.g. table dimensions, list markers).
    """

    index: int
    element_type: NormalizedElementType
    content: str
    page_number: Optional[int] = None
    heading_level: Optional[int] = None
    parent_heading: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedDocument:
    """RAG-owned normalized document representation.

    Fully decoupled from ingestion parsers (such as Docling). Represents the
    document after text cleaning, structural ordering, and context enrichment,
    serving as the direct input to the chunking pipeline.

    Attributes:
        document_id: Unique document identifier.
        file_path: Original file path if ingested from local filesystem.
        format: Document format extension (e.g. 'pdf', 'docx', 'md', 'txt').
        elements: Cleaned, ordered list of NormalizedElement objects.
        text: Synthesized full text of the normalized document.
        metadata: Document-level metadata (file size, page count, element count).
    """

    document_id: str
    file_path: Optional[Path] = None
    format: str = ""
    elements: List[NormalizedElement] = field(default_factory=list)
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def headings(self) -> List[NormalizedElement]:
        """Return all title and section heading elements in document order."""
        return [
            elem
            for elem in self.elements
            if elem.element_type in (NormalizedElementType.TITLE, NormalizedElementType.SECTION_HEADER)
        ]

    @property
    def tables(self) -> List[NormalizedElement]:
        """Return all table elements."""
        return [elem for elem in self.elements if elem.element_type == NormalizedElementType.TABLE]

    @property
    def paragraphs(self) -> List[NormalizedElement]:
        """Return all paragraph and body text elements."""
        return [elem for elem in self.elements if elem.element_type == NormalizedElementType.PARAGRAPH]

    @property
    def list_items(self) -> List[NormalizedElement]:
        """Return all list item elements."""
        return [elem for elem in self.elements if elem.element_type == NormalizedElementType.LIST_ITEM]

    @property
    def page_count(self) -> int:
        """Calculate total page count based on page numbers observed in elements, or metadata."""
        pages = {elem.page_number for elem in self.elements if elem.page_number is not None}
        if pages:
            return max(pages)
        return self.metadata.get("page_count", 1)

    def to_domain_document(self) -> Document:
        """Convert this normalized representation into a domain Document."""
        combined_metadata = {
            **self.metadata,
            "element_count": len(self.elements),
            "heading_count": len(self.headings),
            "table_count": len(self.tables),
            "page_count": self.page_count,
        }
        if self.file_path:
            combined_metadata["source_path"] = str(self.file_path)
        if self.format:
            combined_metadata["format"] = self.format

        return Document(
            id=self.document_id,
            content=self.text,
            metadata=combined_metadata,
        )
