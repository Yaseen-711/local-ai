"""Explicit metadata models and contracts for the RAG subsystem.

Defines typed, structured metadata contracts distinguishing:
1. Document-level metadata (DocumentMetadata)
2. Element-level metadata (ElementMetadata)
3. Provenance metadata (ProvenanceMetadata)
4. Chunk-level and derived metadata (ChunkMetadata)

Ensures metadata remains structured and cleanly decoupled from chunk.content
while maintaining full serializability and backward compatibility with
dict-based access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class DocumentMetadata:
    """Document-level metadata describing source origin, format, and structure.

    Attributes:
        document_id: Unique document identifier.
        source_path: Absolute or relative filesystem path to the source document.
        file_name: Base name of the file (e.g. 'manual.md').
        format: Standardized extension/format (e.g. 'pdf', 'docx', 'md', 'txt').
        file_size_bytes: Size of the raw file in bytes if known.
        title: Extracted or declared document title if available.
        page_count: Total number of pages in the source document.
        element_count: Total count of structural elements parsed from the document.
        custom: Extensible dictionary for source-specific metadata.
    """

    document_id: str
    source_path: Optional[str] = None
    file_name: Optional[str] = None
    format: str = ""
    file_size_bytes: Optional[int] = None
    title: Optional[str] = None
    page_count: int = 1
    element_count: int = 0
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentMetadata to a clean dictionary."""
        d: Dict[str, Any] = {
            "document_id": self.document_id,
            "format": self.format,
            "page_count": self.page_count,
            "element_count": self.element_count,
        }
        if self.source_path is not None:
            d["source_path"] = self.source_path
        if self.file_name is not None:
            d["file_name"] = self.file_name
        if self.file_size_bytes is not None:
            d["file_size_bytes"] = self.file_size_bytes
        if self.title is not None:
            d["title"] = self.title
        if self.custom:
            d["custom"] = dict(self.custom)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentMetadata:
        """Construct DocumentMetadata from a dictionary."""
        known_keys = {
            "document_id",
            "source_path",
            "file_name",
            "format",
            "file_size_bytes",
            "title",
            "page_count",
            "element_count",
            "custom",
        }
        custom = dict(data.get("custom", {}))
        for k, v in data.items():
            if k not in known_keys and k not in custom:
                custom[k] = v

        return cls(
            document_id=data.get("document_id", ""),
            source_path=data.get("source_path"),
            file_name=data.get("file_name"),
            format=data.get("format", ""),
            file_size_bytes=data.get("file_size_bytes"),
            title=data.get("title"),
            page_count=data.get("page_count", 1),
            element_count=data.get("element_count", 0),
            custom=custom,
        )


@dataclass(frozen=True)
class ElementMetadata:
    """Element-level metadata for individual structural units in a document.

    Attributes:
        index: 0-indexed sequential position of the element.
        element_type: Classification (e.g. 'title', 'section_header', 'paragraph', 'table', 'list_item').
        page_number: 1-indexed page number where this element occurs.
        heading_level: Heading level if element is a heading (1 for H1, 2 for H2, etc.).
        parent_heading: Content of the enclosing or most recent preceding heading.
        table_rows: Number of rows if this element is a table.
        table_cols: Number of columns if this element is a table.
        custom: Safe source-specific metadata (e.g. Docling class names).
    """

    index: int
    element_type: str
    page_number: Optional[int] = None
    heading_level: Optional[int] = None
    parent_heading: Optional[str] = None
    table_rows: Optional[int] = None
    table_cols: Optional[int] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ElementMetadata to a clean dictionary."""
        d: Dict[str, Any] = {
            "index": self.index,
            "element_type": self.element_type,
        }
        if self.page_number is not None:
            d["page_number"] = self.page_number
        if self.heading_level is not None:
            d["heading_level"] = self.heading_level
        if self.parent_heading is not None:
            d["parent_heading"] = self.parent_heading
        if self.table_rows is not None:
            d["table_rows"] = self.table_rows
            d["num_rows"] = self.table_rows
        if self.table_cols is not None:
            d["table_cols"] = self.table_cols
            d["num_cols"] = self.table_cols
        if self.custom:
            d["custom"] = dict(self.custom)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ElementMetadata:
        """Construct ElementMetadata from a dictionary."""
        rows = data.get("table_rows") or data.get("num_rows")
        cols = data.get("table_cols") or data.get("num_cols")
        known = {
            "index",
            "element_type",
            "page_number",
            "heading_level",
            "parent_heading",
            "table_rows",
            "num_rows",
            "table_cols",
            "num_cols",
            "custom",
        }
        custom = dict(data.get("custom", {}))
        for k, v in data.items():
            if k not in known and k not in custom:
                custom[k] = v

        return cls(
            index=int(data.get("index", 0)),
            element_type=str(data.get("element_type", "other")),
            page_number=data.get("page_number"),
            heading_level=data.get("heading_level"),
            parent_heading=data.get("parent_heading"),
            table_rows=rows,
            table_cols=cols,
            custom=custom,
        )


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Provenance tracking attributing a chunk back to its source origin and location.

    Attributes:
        document_id: Parent document identifier.
        source_path: File path of the source document.
        file_name: File base name.
        page_numbers: List of 1-indexed pages spanned by this chunk.
        page_range: Formatted page range string (e.g. '1' or '2-3').
        element_indices: Indices of source elements merged into this chunk.
        heading_path: Ancestral heading path breadcrumb list.
        citation: Human-readable citation string suitable for LLM grounding or UI.
    """

    document_id: str
    source_path: Optional[str] = None
    file_name: Optional[str] = None
    page_numbers: List[int] = field(default_factory=list)
    page_range: Optional[str] = None
    element_indices: List[int] = field(default_factory=list)
    heading_path: List[str] = field(default_factory=list)
    citation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProvenanceMetadata to dictionary."""
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "file_name": self.file_name,
            "page_numbers": list(self.page_numbers),
            "page_range": self.page_range,
            "element_indices": list(self.element_indices),
            "heading_path": list(self.heading_path),
            "citation": self.citation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProvenanceMetadata:
        """Reconstruct ProvenanceMetadata from dictionary."""
        return cls(
            document_id=data.get("document_id", ""),
            source_path=data.get("source_path"),
            file_name=data.get("file_name"),
            page_numbers=list(data.get("page_numbers", [])),
            page_range=data.get("page_range"),
            element_indices=list(data.get("element_indices", [])),
            heading_path=list(data.get("heading_path", [])),
            citation=data.get("citation", ""),
        )


@dataclass(frozen=True)
class ChunkMetadata:
    """Strongly-typed metadata model for a RAG chunk.

    Contains core identifiers, structural and hierarchical context, provenance,
    derived indicators (table, list, code, page range), and split information.
    """

    # Core identifiers
    chunk_id: str
    document_id: str
    chunk_index: int

    # Structural hierarchy
    heading: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)

    # Document & file context
    source_path: Optional[str] = None
    file_name: Optional[str] = None
    format: str = ""
    document_title: Optional[str] = None

    # Element attribution and pages
    element_indices: List[int] = field(default_factory=list)
    element_types: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)

    # Derived fields
    primary_page: Optional[int] = None
    page_range: Optional[str] = None
    primary_element_type: str = "paragraph"
    has_table: bool = False
    has_list: bool = False
    has_code: bool = False
    is_table: bool = False
    table_rows: Optional[int] = None
    table_cols: Optional[int] = None

    # Split provenance
    is_split: bool = False
    split_part: Optional[int] = None
    total_parts: Optional[int] = None

    # Citation
    citation: str = ""

    # Extension dictionary for domain-specific tags
    custom: Dict[str, Any] = field(default_factory=dict)

    def get_provenance(self) -> ProvenanceMetadata:
        """Extract dedicated ProvenanceMetadata object for this chunk."""
        return ProvenanceMetadata(
            document_id=self.document_id,
            source_path=self.source_path,
            file_name=self.file_name,
            page_numbers=list(self.page_numbers),
            page_range=self.page_range,
            element_indices=list(self.element_indices),
            heading_path=list(self.heading_path),
            citation=self.citation,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ChunkMetadata to dictionary preserving all backward-compatible keys."""
        d: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "heading": self.heading,
            "heading_path": list(self.heading_path),
            "element_indices": list(self.element_indices),
            "element_types": list(self.element_types),
            "page_numbers": list(self.page_numbers),
            "primary_page": self.primary_page,
            "page_range": self.page_range,
            "primary_element_type": self.primary_element_type,
            "has_table": self.has_table,
            "has_list": self.has_list,
            "has_code": self.has_code,
            "is_table": self.is_table,
            "is_split": self.is_split,
            "citation": self.citation,
        }

        if self.source_path is not None:
            d["source_path"] = self.source_path
        if self.file_name is not None:
            d["file_name"] = self.file_name
        if self.format:
            d["format"] = self.format
        if self.document_title is not None:
            d["document_title"] = self.document_title
        if self.split_part is not None:
            d["split_part"] = self.split_part
        if self.total_parts is not None:
            d["total_parts"] = self.total_parts
        if self.table_rows is not None:
            d["table_rows"] = self.table_rows
            d["num_rows"] = self.table_rows
        if self.table_cols is not None:
            d["table_cols"] = self.table_cols
            d["num_cols"] = self.table_cols

        if self.custom:
            for k, v in self.custom.items():
                if k not in d:
                    d[k] = v

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChunkMetadata:
        """Reconstruct a ChunkMetadata dataclass from a metadata dictionary."""
        known_keys = {
            "chunk_id",
            "document_id",
            "chunk_index",
            "heading",
            "heading_path",
            "source_path",
            "file_name",
            "format",
            "document_title",
            "element_indices",
            "element_types",
            "page_numbers",
            "primary_page",
            "page_range",
            "primary_element_type",
            "has_table",
            "has_list",
            "has_code",
            "is_table",
            "table_rows",
            "num_rows",
            "table_cols",
            "num_cols",
            "is_split",
            "split_part",
            "total_parts",
            "citation",
            "custom",
        }

        table_rows = data.get("table_rows") or data.get("num_rows")
        table_cols = data.get("table_cols") or data.get("num_cols")

        custom = dict(data.get("custom", {}))
        for k, v in data.items():
            if k not in known_keys and k not in custom:
                custom[k] = v

        return cls(
            chunk_id=str(data.get("chunk_id", "")),
            document_id=str(data.get("document_id", "")),
            chunk_index=int(data.get("chunk_index", 0)),
            heading=data.get("heading"),
            heading_path=list(data.get("heading_path", [])),
            source_path=data.get("source_path"),
            file_name=data.get("file_name"),
            format=str(data.get("format", "")),
            document_title=data.get("document_title"),
            element_indices=list(data.get("element_indices", [])),
            element_types=list(data.get("element_types", [])),
            page_numbers=list(data.get("page_numbers", [])),
            primary_page=data.get("primary_page"),
            page_range=data.get("page_range"),
            primary_element_type=str(data.get("primary_element_type", "paragraph")),
            has_table=bool(data.get("has_table", False)),
            has_list=bool(data.get("has_list", False)),
            has_code=bool(data.get("has_code", False)),
            is_table=bool(data.get("is_table", False)),
            table_rows=table_rows,
            table_cols=table_cols,
            is_split=bool(data.get("is_split", False)),
            split_part=data.get("split_part"),
            total_parts=data.get("total_parts"),
            citation=str(data.get("citation", "")),
            custom=custom,
        )

    @classmethod
    def from_chunk(cls, chunk: Any) -> ChunkMetadata:
        """Construct ChunkMetadata from a domain models.Chunk instance."""
        meta_dict = dict(getattr(chunk, "metadata", {}))
        if "chunk_id" not in meta_dict and hasattr(chunk, "id"):
            meta_dict["chunk_id"] = chunk.id
        if "document_id" not in meta_dict and hasattr(chunk, "document_id"):
            meta_dict["document_id"] = chunk.document_id
        return cls.from_dict(meta_dict)

    @classmethod
    def build(
        cls,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        elements: Sequence[Any],
        heading: Optional[str] = None,
        heading_path: Optional[List[str]] = None,
        source_path: Optional[str] = None,
        format: str = "",
        document_title: Optional[str] = None,
        is_split: bool = False,
        split_part: Optional[int] = None,
        total_parts: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> ChunkMetadata:
        """Factory method to compute and derive a complete ChunkMetadata instance."""
        extra = dict(extra_meta or {})

        elem_indices = [getattr(e, "index") for e in elements if hasattr(e, "index")]

        # Determine element types preserving order
        raw_types = []
        for e in elements:
            etype = getattr(e, "element_type", None)
            if hasattr(etype, "value"):
                raw_types.append(etype.value)
            elif isinstance(etype, str):
                raw_types.append(etype)
        elem_types = list(dict.fromkeys(raw_types))

        # Determine page numbers
        pages_set = {
            getattr(e, "page_number")
            for e in elements
            if getattr(e, "page_number", None) is not None
        }
        page_numbers = sorted(list(pages_set))

        primary_page = page_numbers[0] if page_numbers else None
        if not page_numbers:
            page_range = None
        elif len(page_numbers) == 1:
            page_range = str(page_numbers[0])
        else:
            page_range = f"{page_numbers[0]}-{page_numbers[-1]}"

        # Structural flags
        has_table = "table" in elem_types or bool(extra.get("is_table"))
        has_list = "list_item" in elem_types
        has_code = "code" in elem_types
        is_table = bool(extra.get("is_table")) or (len(elem_types) == 1 and elem_types[0] == "table")

        # Table dimensions
        table_rows = extra.get("table_rows") or extra.get("num_rows")
        table_cols = extra.get("table_cols") or extra.get("num_cols")
        if table_rows is None or table_cols is None:
            for e in elements:
                emeta = getattr(e, "metadata", {})
                if table_rows is None and ("num_rows" in emeta or "table_rows" in emeta):
                    table_rows = emeta.get("table_rows") or emeta.get("num_rows")
                if table_cols is None and ("num_cols" in emeta or "table_cols" in emeta):
                    table_cols = emeta.get("table_cols") or emeta.get("num_cols")

        # Primary element type
        if is_table or has_table:
            primary_type = "table"
        elif has_code:
            primary_type = "code"
        elif has_list and all(t == "list_item" for t in elem_types):
            primary_type = "list_item"
        elif elem_types:
            primary_type = elem_types[0]
        else:
            primary_type = "paragraph"

        file_name = Path(source_path).name if source_path else None

        # Build citation string
        citation_src = file_name or document_title or document_id
        citation_parts = [citation_src]
        if page_range:
            citation_parts.append(f"p. {page_range}")
        if heading:
            citation_parts.append(f"({heading})")
        citation = " ".join(citation_parts)

        h_path = list(heading_path) if heading_path else ([] if not heading else [heading])

        # Separate custom/unknown keys
        known = {
            "chunk_id",
            "document_id",
            "chunk_index",
            "heading",
            "heading_path",
            "source_path",
            "file_name",
            "format",
            "document_title",
            "element_indices",
            "element_types",
            "page_numbers",
            "primary_page",
            "page_range",
            "primary_element_type",
            "has_table",
            "has_list",
            "has_code",
            "is_table",
            "table_rows",
            "num_rows",
            "table_cols",
            "num_cols",
            "is_split",
            "split_part",
            "total_parts",
            "citation",
        }
        custom = {k: v for k, v in extra.items() if k not in known}

        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            heading=heading,
            heading_path=h_path,
            source_path=source_path,
            file_name=file_name,
            format=format,
            document_title=document_title,
            element_indices=elem_indices,
            element_types=elem_types,
            page_numbers=page_numbers,
            primary_page=primary_page,
            page_range=page_range,
            primary_element_type=primary_type,
            has_table=has_table,
            has_list=has_list,
            has_code=has_code,
            is_table=is_table,
            table_rows=table_rows,
            table_cols=table_cols,
            is_split=is_split,
            split_part=split_part,
            total_parts=total_parts,
            citation=citation,
            custom=custom,
        )

    def merge_elements(
        self,
        new_elements: Sequence[Any],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> ChunkMetadata:
        """Return a new ChunkMetadata with newly merged elements included."""
        extra = dict(extra_meta or {})

        new_indices = [getattr(e, "index") for e in new_elements if hasattr(e, "index")]
        combined_indices = self.element_indices + new_indices

        new_types = []
        for e in new_elements:
            etype = getattr(e, "element_type", None)
            if hasattr(etype, "value"):
                new_types.append(etype.value)
            elif isinstance(etype, str):
                new_types.append(etype)
        combined_types = list(dict.fromkeys(self.element_types + new_types))

        new_pages = {
            getattr(e, "page_number")
            for e in new_elements
            if getattr(e, "page_number", None) is not None
        }
        combined_pages = sorted(list(set(self.page_numbers).union(new_pages)))

        primary_page = combined_pages[0] if combined_pages else None
        if not combined_pages:
            page_range = None
        elif len(combined_pages) == 1:
            page_range = str(combined_pages[0])
        else:
            page_range = f"{combined_pages[0]}-{combined_pages[-1]}"

        has_table = self.has_table or "table" in combined_types or bool(extra.get("is_table"))
        has_list = self.has_list or "list_item" in combined_types
        has_code = self.has_code or "code" in combined_types
        is_table = self.is_table or bool(extra.get("is_table"))

        # Primary element type
        if is_table or has_table:
            primary_type = "table"
        elif has_code:
            primary_type = "code"
        elif has_list and all(t == "list_item" for t in combined_types):
            primary_type = "list_item"
        elif combined_types:
            primary_type = combined_types[0]
        else:
            primary_type = "paragraph"

        citation_src = self.file_name or self.document_title or self.document_id
        citation_parts = [citation_src]
        if page_range:
            citation_parts.append(f"p. {page_range}")
        if self.heading:
            citation_parts.append(f"({self.heading})")
        citation = " ".join(citation_parts)

        return ChunkMetadata(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            chunk_index=self.chunk_index,
            heading=self.heading,
            heading_path=list(self.heading_path),
            source_path=self.source_path,
            file_name=self.file_name,
            format=self.format,
            document_title=self.document_title,
            element_indices=combined_indices,
            element_types=combined_types,
            page_numbers=combined_pages,
            primary_page=primary_page,
            page_range=page_range,
            primary_element_type=primary_type,
            has_table=has_table,
            has_list=has_list,
            has_code=has_code,
            is_table=is_table,
            table_rows=self.table_rows or extra.get("num_rows"),
            table_cols=self.table_cols or extra.get("num_cols"),
            is_split=self.is_split,
            split_part=self.split_part,
            total_parts=self.total_parts,
            citation=citation,
            custom={**self.custom, **extra},
        )
