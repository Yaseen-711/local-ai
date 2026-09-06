"""Standard document normalizer implementation for RAG subsystem.

Processes IngestedDocument objects into clean, deterministic NormalizedDocument
representations, preserving document structure, heading hierarchies, tables,
lists, and page numbers.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

from rag.ingestion.models import DocumentElement, ElementType, IngestedDocument
from rag.normalization.interfaces import DocumentNormalizer
from rag.normalization.models import (
    NormalizedDocument,
    NormalizedElement,
    NormalizedElementType,
)

# Map ingestion element types to normalized element types
_ELEMENT_TYPE_MAP: Dict[ElementType, NormalizedElementType] = {
    ElementType.TITLE: NormalizedElementType.TITLE,
    ElementType.SECTION_HEADER: NormalizedElementType.SECTION_HEADER,
    ElementType.PARAGRAPH: NormalizedElementType.PARAGRAPH,
    ElementType.LIST_ITEM: NormalizedElementType.LIST_ITEM,
    ElementType.TABLE: NormalizedElementType.TABLE,
    ElementType.CODE: NormalizedElementType.CODE,
    ElementType.OTHER: NormalizedElementType.OTHER,
}

_MULTI_SPACE_REGEX = re.compile(r"[ \t]+")
_MULTI_NEWLINE_REGEX = re.compile(r"\n{3,}")


def clean_text_whitespace(text: str) -> str:
    """Normalize whitespace in standard text blocks without destroying line breaks.

    - Normalizes unicode (NFKC) converting non-breaking spaces and irregular characters.
    - Standardizes CRLF to LF.
    - Collapses multiple horizontal spaces/tabs on each line to a single space.
    - Collapses 3 or more consecutive newlines into 2 (paragraph break).
    - Strips leading and trailing outer whitespace.
    """
    if not text:
        return ""

    # Unicode standardization
    normalized = unicodedata.normalize("NFKC", text)
    # Line endings standardization
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize horizontal whitespace line by line to preserve line breaks
    lines = [_MULTI_SPACE_REGEX.sub(" ", line).strip() for line in normalized.split("\n")]
    result = "\n".join(lines)

    # Collapse excessive vertical spacing
    result = _MULTI_NEWLINE_REGEX.sub("\n\n", result)
    return result.strip()


def clean_table_content(table_text: str) -> str:
    """Normalize table markdown content while strictly preserving row/column structure."""
    if not table_text:
        return ""

    normalized = unicodedata.normalize("NFKC", table_text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: List[str] = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if stripped:
            # Normalize redundant spaces between pipes while preserving table row structure
            line_clean = _MULTI_SPACE_REGEX.sub(" ", stripped)
            cleaned_lines.append(line_clean)

    return "\n".join(cleaned_lines)


class StandardDocumentNormalizer(DocumentNormalizer):
    """Production implementation of the DocumentNormalizer protocol.

    Performs:
    1. Unicode & whitespace normalization.
    2. Empty element pruning.
    3. Structural hierarchy tracking (associating parent headings to sections).
    4. Table and list structural preservation.
    5. Deterministic element re-indexing.
    6. Complete decoupling from Docling data structures.
    """

    def __init__(
        self,
        strip_empty: bool = True,
        normalize_unicode: bool = True,
        track_hierarchy: bool = True,
    ) -> None:
        self.strip_empty = strip_empty
        self.normalize_unicode = normalize_unicode
        self.track_hierarchy = track_hierarchy

    def normalize(self, document: IngestedDocument) -> NormalizedDocument:
        """Normalize an IngestedDocument into a clean NormalizedDocument.

        Args:
            document: Raw IngestedDocument from document ingestion.

        Returns:
            NormalizedDocument with cleaned elements and preserved structural context.
        """
        normalized_elements: List[NormalizedElement] = []
        current_parent_heading: Optional[str] = None
        element_index = 0

        for raw_elem in document.elements:
            norm_type = _ELEMENT_TYPE_MAP.get(
                raw_elem.element_type,
                NormalizedElementType.OTHER,
            )

            # Choose appropriate cleaning rule depending on element type
            if norm_type == NormalizedElementType.TABLE:
                cleaned_content = clean_table_content(raw_elem.content)
            else:
                cleaned_content = clean_text_whitespace(raw_elem.content)

            # Handle empty elements
            if self.strip_empty and not cleaned_content:
                continue

            # Track heading hierarchy
            if norm_type in (NormalizedElementType.TITLE, NormalizedElementType.SECTION_HEADER):
                if self.track_hierarchy:
                    current_parent_heading = cleaned_content

            norm_elem = NormalizedElement(
                index=element_index,
                element_type=norm_type,
                content=cleaned_content,
                page_number=raw_elem.page_number,
                heading_level=raw_elem.heading_level,
                parent_heading=current_parent_heading if norm_type not in (
                    NormalizedElementType.TITLE,
                    NormalizedElementType.SECTION_HEADER,
                ) else None,
                metadata=dict(raw_elem.metadata),
            )
            normalized_elements.append(norm_elem)
            element_index += 1

        # Reconstruct synthesized full text from normalized elements
        synthesized_text = "\n\n".join(elem.content for elem in normalized_elements if elem.content)

        # Discover document title from elements if not already present
        doc_title: Optional[str] = document.metadata.get("title")
        if not doc_title:
            for elem in normalized_elements:
                if elem.element_type == NormalizedElementType.TITLE and elem.content.strip():
                    doc_title = elem.content.strip()
                    break

        # Retain document metadata with normalization records
        combined_metadata = {
            **document.metadata,
            "normalized": True,
            "normalized_element_count": len(normalized_elements),
            "raw_element_count": len(document.elements),
        }
        if doc_title and "title" not in combined_metadata:
            combined_metadata["title"] = doc_title
        if document.file_path and "source_path" not in combined_metadata:
            combined_metadata["source_path"] = str(document.file_path)
        if document.format and "format" not in combined_metadata:
            combined_metadata["format"] = document.format


        return NormalizedDocument(
            document_id=document.id,
            file_path=document.file_path,
            format=document.format,
            elements=normalized_elements,
            text=synthesized_text,
            metadata=combined_metadata,
        )
