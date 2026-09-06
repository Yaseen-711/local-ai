"""Normalized Document Representation data contracts.

These contracts provide a clean, high-fidelity, decoupled document representation
with page, table, and bounding-box provenance.

Strict Invariant:
  This module contains PURE Python data structures. It must NEVER import or reference
  Docling internal types (such as DoclingDocument or DocItem).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Normalized spatial coordinates for layout and visual elements.

    Attributes:
        l: Left coordinate.
        t: Top coordinate.
        r: Right coordinate.
        b: Bottom coordinate.
        coord_origin: Coordinate system origin hint (e.g. 'TOPLEFT', 'BOTTOMLEFT').
    """
    l: float
    t: float
    r: float
    b: float
    coord_origin: str = "BOTTOMLEFT"

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.l, self.t, self.r, self.b)


@dataclass(frozen=True)
class Provenance:
    """Lightweight origin and spatial location descriptor for extracted document items.

    Attributes:
        page_number: 1-indexed page where this element appears.
        bbox: Optional spatial bounding box.
        char_start: Optional starting character offset in document text.
        char_end: Optional ending character offset in document text.
    """
    page_number: int
    bbox: Optional[BoundingBox] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


@dataclass(frozen=True)
class DocumentTable:
    """Structured tabular data extracted from a document.

    Attributes:
        table_id: Unique identifier for this table within the document.
        page_number: 1-indexed page number where table begins.
        num_rows: Total row count.
        num_cols: Total column count.
        grid: 2D array of string cells [rows][cols].
        markdown: Clean GitHub Flavored Markdown table representation.
        provenance: Spatial provenance on page.
    """
    table_id: str
    page_number: int
    num_rows: int
    num_cols: int
    grid: List[List[str]] = field(default_factory=list)
    markdown: str = ""
    provenance: Optional[Provenance] = None


@dataclass(frozen=True)
class DocumentFigure:
    """Extracted figure, chart, or image element.

    Attributes:
        figure_id: Unique figure identifier.
        page_number: 1-indexed page number.
        caption: Optional descriptive caption.
        provenance: Spatial bounding box and location.
        artifact_id: Optional reference ID if image binary was saved to artifact storage.
    """
    figure_id: str
    page_number: int
    caption: Optional[str] = None
    provenance: Optional[Provenance] = None
    artifact_id: Optional[str] = None


@dataclass(frozen=True)
class DocumentPage:
    """Single page representation.

    Attributes:
        page_number: 1-indexed page number.
        text: Plain or markdown text content of this page.
        dimension: Optional (width, height) in points.
        tables: Tables located on this page.
        figures: Figures located on this page.
    """
    page_number: int
    text: str = ""
    dimension: Optional[Tuple[float, float]] = None
    tables: List[DocumentTable] = field(default_factory=list)
    figures: List[DocumentFigure] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizedDocument:
    """Canonical normalized document representation.

    Represents the output of document parsing and understanding without exposing
    any parser backend specifics (e.g. Docling internals). Downstream tasks can
    consume `text`, `tables`, or `pages` via standard input references.

    Attributes:
        document_id: Unique identifier for this document instance.
        filename: Source filename or basename.
        mime_type: Detected or specified MIME type.
        text: Unified full markdown text of the document.
        page_count: Total number of pages.
        pages: Ordered list of document pages.
        tables: All structured tables extracted from the document.
        figures: All visual figures/images extracted.
        metadata: Extraction metadata (has_selectable_text, ocr_applied, parser_backend, etc.).
    """
    document_id: str
    filename: str
    mime_type: str
    text: str
    page_count: int
    pages: List[DocumentPage] = field(default_factory=list)
    tables: List[DocumentTable] = field(default_factory=list)
    figures: List[DocumentFigure] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sha256_checksum(self) -> str:
        """SHA-256 checksum of source document."""
        return str(self.metadata.get("sha256", ""))

    @property
    def file_size_bytes(self) -> int:
        """File size in bytes of source document."""
        return int(self.metadata.get("file_size_bytes", 0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary for TaskResult.output."""
        return asdict(self)


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizedDocument":
        """Reconstruct NormalizedDocument from a serialized dictionary."""
        pages = []
        for p in data.get("pages", []):
            tables = [
                DocumentTable(
                    table_id=t["table_id"],
                    page_number=t["page_number"],
                    num_rows=t["num_rows"],
                    num_cols=t["num_cols"],
                    grid=t.get("grid", []),
                    markdown=t.get("markdown", ""),
                    provenance=Provenance(**t["provenance"]) if t.get("provenance") else None,
                )
                for t in p.get("tables", [])
            ]
            figures = [
                DocumentFigure(
                    figure_id=f["figure_id"],
                    page_number=f["page_number"],
                    caption=f.get("caption"),
                    provenance=Provenance(**f["provenance"]) if f.get("provenance") else None,
                    artifact_id=f.get("artifact_id"),
                )
                for f in p.get("figures", [])
            ]
            dim = tuple(p["dimension"]) if p.get("dimension") else None
            pages.append(
                DocumentPage(
                    page_number=p["page_number"],
                    text=p.get("text", ""),
                    dimension=dim,  # type: ignore[arg-type]
                    tables=tables,
                    figures=figures,
                )
            )

        tables = [
            DocumentTable(
                table_id=t["table_id"],
                page_number=t["page_number"],
                num_rows=t["num_rows"],
                num_cols=t["num_cols"],
                grid=t.get("grid", []),
                markdown=t.get("markdown", ""),
                provenance=Provenance(**t["provenance"]) if t.get("provenance") else None,
            )
            for t in data.get("tables", [])
        ]

        figures = [
            DocumentFigure(
                figure_id=f["figure_id"],
                page_number=f["page_number"],
                caption=f.get("caption"),
                provenance=Provenance(**f["provenance"]) if f.get("provenance") else None,
                artifact_id=f.get("artifact_id"),
            )
            for f in data.get("figures", [])
        ]

        return cls(
            document_id=data["document_id"],
            filename=data["filename"],
            mime_type=data["mime_type"],
            text=data["text"],
            page_count=data["page_count"],
            pages=pages,
            tables=tables,
            figures=figures,
            metadata=dict(data.get("metadata", {})),
        )
