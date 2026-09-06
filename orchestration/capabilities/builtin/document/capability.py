"""Built-in Document Understanding capability.

Ingests document files via reference (URI or path), runs layout and OCR parsing
behind the DocumentParser seam (Docling or fallback), and outputs a NormalizedDocument.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.document.base import (
    DocumentParseOptions,
    DocumentParser,
)
from orchestration.capabilities.builtin.document.fallback_parser import FallbackDocumentParser
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskResult

logger = logging.getLogger(__name__)


def _resolve_file_path(target: Any) -> Path:
    """Resolve a path or URI reference to a canonical local Path."""
    if isinstance(target, Path):
        return target.resolve()

    if isinstance(target, DataReference):
        target = target.uri or target.key

    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"Invalid document input: expected path or URI, got {type(target).__name__}")

    raw = target.strip()
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        raw = unquote(parsed.path)

    path = Path(raw).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Document target is not a file: {path}")

    return path


class DocumentUnderstandingCapability:
    """Capability executing document understanding, layout parsing, and OCR.

    Semantic contract:
        Inputs / Parameters:
            - 'document_uri' (str) OR 'file_path' (str) OR 'document' (DataReference):
              Reference to the document file on disk.
            - 'do_ocr' (bool, default: True): Whether to run OCR on scanned/image pages.
            - 'extract_tables' (bool, default: True): Whether to extract structured tables.
            - 'extract_figures' (bool, default: False): Whether to extract visual figures.
            - 'max_pages' (int, optional): Page processing limit.
            - 'force_fallback' (bool, default: False): Force use of lightweight fallback parser.
    """

    def __init__(self, parser: Optional[DocumentParser] = None) -> None:
        """Initialize capability with an optional DocumentParser.

        If parser is None, attempts to use DoclingDocumentParser if docling is installed,
        falling back to FallbackDocumentParser.
        """
        self._parser = parser

    @property
    def capability_id(self) -> str:
        return "document.understand"

    def _get_parser(self, force_fallback: bool = False) -> DocumentParser:
        if self._parser is not None and not force_fallback:
            return self._parser

        if force_fallback:
            return FallbackDocumentParser()

        # Try Docling if available
        try:
            from orchestration.capabilities.builtin.document.docling_parser import (
                DoclingDocumentParser,
            )
            return DoclingDocumentParser()
        except ImportError:
            logger.info("Docling not available; using FallbackDocumentParser.")
            return FallbackDocumentParser()

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        # 1. Resolve file reference (do not accept large raw byte dumps)
        target = (
            inputs.get("document")
            or inputs.get("document_uri")
            or inputs.get("file_path")
            or parameters.get("file_path")
            or parameters.get("document_uri")
        )

        if target is None:
            raise ValueError(
                f"Capability '{self.capability_id}' requires a document file reference "
                f"('document', 'document_uri', or 'file_path') in inputs or parameters."
            )

        if isinstance(target, (bytes, bytearray)):
            raise ValueError(
                f"Capability '{self.capability_id}' rejects raw byte payloads in parameters. "
                "Provide a file URI or DataReference instead."
            )

        file_path = _resolve_file_path(target)

        # 2. Extract options
        do_ocr = bool(parameters.get("do_ocr", inputs.get("do_ocr", True)))
        extract_tables = bool(parameters.get("extract_tables", inputs.get("extract_tables", True)))
        extract_figures = bool(parameters.get("extract_figures", inputs.get("extract_figures", False)))
        max_pages = parameters.get("max_pages") or inputs.get("max_pages")
        if max_pages is not None:
            max_pages = int(max_pages)

        force_fallback = bool(parameters.get("force_fallback", inputs.get("force_fallback", False)))

        options = DocumentParseOptions(
            do_ocr=do_ocr,
            extract_tables=extract_tables,
            extract_figures=extract_figures,
            max_pages=max_pages,
        )

        # 3. Parse document
        parser = self._get_parser(force_fallback=force_fallback)
        normalized_doc = parser.parse(file_path=file_path, options=options)

        # 4. Build output and references
        doc_dict = normalized_doc.to_dict()
        file_uri = file_path.as_uri()

        references = [
            DataReference(
                key="text",
                uri=file_uri,
                mime_type="text/markdown",
                metadata={"page_count": normalized_doc.page_count},
            ),
            DataReference(
                key="tables",
                uri=file_uri,
                mime_type="application/json",
                metadata={"table_count": len(normalized_doc.tables)},
            ),
        ]

        return TaskResult(
            output=doc_dict,
            references=references,
            metadata=dict(normalized_doc.metadata),
        )
