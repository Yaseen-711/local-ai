"""Docling document ingestion adapter.

Encapsulates Docling's DocumentConverter behind the project-owned DocumentIngester
protocol. Extracts headings, paragraphs, tables, lists, and page numbers into
the structured IngestedDocument representation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from rag.ingestion.errors import DocumentParsingError, UnsupportedDocumentError
from rag.ingestion.interfaces import DocumentIngester
from rag.ingestion.models import DocumentElement, ElementType, IngestedDocument

SUPPORTED_EXTENSIONS: Set[str] = frozenset({
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".md",
    ".asciidoc",
    ".csv",
    ".txt",
})


class DoclingDocumentIngester(DocumentIngester):
    """Concrete DocumentIngester leveraging IBM Docling.

    Isolates all third-party Docling data types and runtime logic. Emits structured
    IngestedDocument models without performing chunking, embedding, or indexing.
    """

    def __init__(
        self,
        do_ocr: bool = False,
        converter: Optional[Any] = None,
    ) -> None:
        """Initialize the Docling document ingester.

        Args:
            do_ocr: Whether to perform OCR on scanned pages (default: False for fast local processing).
            converter: Optional custom/mock DocumentConverter instance for dependency injection in tests.
        """
        self.do_ocr = do_ocr
        self._converter = converter

    def _get_converter(self) -> Any:
        """Lazily initialize DocumentConverter with configured format options."""
        if self._converter is not None:
            return self._converter

        from rag.offline import (
            ensure_offline_environment,
            get_expected_model_path,
            is_offline_mode,
            OfflineModelNotFoundError,
        )

        ensure_offline_environment()

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.do_ocr
            # Explicitly disable remote parsing / remote OCR services to guarantee offline safety
            pipeline_options.enable_remote_services = False

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            return self._converter
        except Exception as exc:
            if is_offline_mode() and any(
                term in str(exc).lower() for term in ("connect", "offline", "network", "download", "http")
            ):
                raise OfflineModelNotFoundError(
                    model_name="docling-project/docling-models",
                    component="DoclingDocumentIngester",
                    expected_location=str(get_expected_model_path("docling-project/docling-models")),
                    details=str(exc),
                ) from exc
            raise DocumentParsingError(
                f"Failed to initialize Docling DocumentConverter: {exc}"
            ) from exc

    def supports_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file extension is among verified Docling-supported formats."""
        path = Path(file_path)
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    def ingest(self, file_path: Union[str, Path]) -> IngestedDocument:
        """Ingest a local document using Docling and produce a structured IngestedDocument.

        Args:
            file_path: Absolute or relative path to a local document file.

        Returns:
            Structured IngestedDocument preserving headings, paragraphs, tables, lists, and pages.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file_path points to a directory.
            UnsupportedDocumentError: If the file format is not supported.
            DocumentParsingError: If Docling fails during conversion.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        if path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {path}")

        if not self.supports_format(path):
            raise UnsupportedDocumentError(
                f"Unsupported document format '{path.suffix}'. "
                f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        converter = self._get_converter()

        try:
            conversion_result = converter.convert(path)
            docling_doc = conversion_result.document
        except Exception as exc:
            raise DocumentParsingError(
                f"Docling conversion failed for '{path.name}': {exc}"
            ) from exc

        # Extract structured elements
        elements = self._extract_elements(docling_doc)

        # Synthesize full text / markdown representation
        try:
            full_text = docling_doc.export_to_markdown()
        except Exception:
            full_text = "\n\n".join(elem.content for elem in elements if elem.content)

        # Compute deterministic ID based on file path and content hash
        file_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:12]
        doc_id = f"doc_{path.stem}_{file_hash}"

        file_stat = path.stat()
        page_count = len(docling_doc.pages) if hasattr(docling_doc, "pages") else 1

        metadata: Dict[str, Any] = {
            "file_name": path.name,
            "file_size_bytes": file_stat.st_size,
            "format": path.suffix.lstrip(".").lower(),
            "page_count": page_count,
            "element_count": len(elements),
        }

        return IngestedDocument(
            id=doc_id,
            file_path=path,
            format=path.suffix.lstrip(".").lower(),
            elements=elements,
            text=full_text,
            metadata=metadata,
        )

    def _extract_elements(self, docling_doc: Any) -> List[DocumentElement]:
        """Map Docling internal document items to RAG DocumentElement instances."""
        elements: List[DocumentElement] = []

        if not hasattr(docling_doc, "iterate_items"):
            return elements

        for item, level in docling_doc.iterate_items():
            cls_name = type(item).__name__

            # Extract provenance / page number if available
            page_no: Optional[int] = None
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0 and hasattr(prov[0], "page_no"):
                page_no = prov[0].page_no

            # 1. Titles
            if cls_name == "TitleItem":
                text = getattr(item, "text", "") or ""
                if text.strip():
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.TITLE,
                            content=text.strip(),
                            page_number=page_no,
                            heading_level=1,
                        )
                    )

            # 2. Section Headings
            elif cls_name == "SectionHeaderItem":
                text = getattr(item, "text", "") or ""
                h_level = getattr(item, "level", level)
                if text.strip():
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.SECTION_HEADER,
                            content=text.strip(),
                            page_number=page_no,
                            heading_level=h_level,
                        )
                    )

            # 3. Tables
            elif cls_name == "TableItem":
                try:
                    table_md = item.export_to_markdown(doc=docling_doc)
                except Exception:
                    table_md = getattr(item, "text", "") or ""

                table_meta: Dict[str, Any] = {}
                data = getattr(item, "data", None)
                if data is not None:
                    if hasattr(data, "num_rows"):
                        table_meta["num_rows"] = data.num_rows
                    if hasattr(data, "num_cols"):
                        table_meta["num_cols"] = data.num_cols

                elements.append(
                    DocumentElement(
                        element_type=ElementType.TABLE,
                        content=table_md,
                        page_number=page_no,
                        metadata=table_meta,
                    )
                )

            # 4. List Items
            elif cls_name == "ListItem":
                text = getattr(item, "text", "") or ""
                if text.strip():
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.LIST_ITEM,
                            content=text.strip(),
                            page_number=page_no,
                        )
                    )

            # 5. Standard Text / Paragraphs
            elif cls_name == "TextItem":
                text = getattr(item, "text", "") or ""
                if text.strip():
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.PARAGRAPH,
                            content=text.strip(),
                            page_number=page_no,
                        )
                    )

            # 6. Other / Code / Unclassified
            else:
                text = getattr(item, "text", None)
                if text and text.strip():
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.OTHER,
                            content=text.strip(),
                            page_number=page_no,
                            metadata={"docling_class": cls_name},
                        )
                    )

        return elements
