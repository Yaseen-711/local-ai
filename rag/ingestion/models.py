"""Structured document representation produced by the ingestion layer.

Preserves rich structural elements (headings, paragraphs, tables, lists, pages)
prior to normalization, chunking, or embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.domain.models import Document


class ElementType(str, Enum):
    """Semantic type of an ingested document element."""

    TITLE = "title"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    OTHER = "other"


@dataclass(frozen=True)
class DocumentElement:
    """A structural unit inside an ingested document (e.g. heading, paragraph, table)."""

    element_type: ElementType
    content: str
    page_number: Optional[int] = None
    heading_level: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestedDocument:
    """Structured intermediate document representation produced by an ingester.

    Contains the parsed document metadata and ordered structural elements
    before downstream normalization and chunking.
    """

    id: str
    file_path: Path
    format: str
    elements: List[DocumentElement]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def headings(self) -> List[DocumentElement]:
        """Return all title and section heading elements in document order."""
        return [
            elem
            for elem in self.elements
            if elem.element_type in (ElementType.TITLE, ElementType.SECTION_HEADER)
        ]

    @property
    def tables(self) -> List[DocumentElement]:
        """Return all table elements."""
        return [elem for elem in self.elements if elem.element_type == ElementType.TABLE]

    @property
    def paragraphs(self) -> List[DocumentElement]:
        """Return all paragraph and text body elements."""
        return [elem for elem in self.elements if elem.element_type == ElementType.PARAGRAPH]

    @property
    def list_items(self) -> List[DocumentElement]:
        """Return all list item elements."""
        return [elem for elem in self.elements if elem.element_type == ElementType.LIST_ITEM]

    def to_domain_document(self) -> Document:
        """Convert this structured representation into a domain Document.

        Useful for downstream stages that accept standard domain Documents.
        """
        combined_metadata = {
            **self.metadata,
            "source_path": str(self.file_path),
            "format": self.format,
            "element_count": len(self.elements),
            "table_count": len(self.tables),
            "heading_count": len(self.headings),
        }
        return Document(
            id=self.id,
            content=self.text,
            metadata=combined_metadata,
        )
