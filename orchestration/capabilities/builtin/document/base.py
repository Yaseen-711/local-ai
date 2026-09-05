"""Document parser protocol and parse options.

Defines the pluggable seam between the DocumentUnderstandingCapability
and concrete document parsing engines (Docling, lightweight fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from orchestration.capabilities.builtin.document.types import NormalizedDocument


@dataclass(frozen=True)
class DocumentParseOptions:
    """Options governing document parsing behavior.

    Attributes:
        do_ocr: Whether to run OCR for image-based/scanned pages.
        extract_tables: Whether to run structured table extraction.
        extract_figures: Whether to extract figure metadata.
        max_pages: Optional upper limit on pages to parse.
    """
    do_ocr: bool = True
    extract_tables: bool = True
    extract_figures: bool = False
    max_pages: Optional[int] = None


@runtime_checkable
class DocumentParser(Protocol):
    """Protocol for document parsing and structure extraction engines."""

    def parse(
        self,
        file_path: Path,
        options: Optional[DocumentParseOptions] = None,
    ) -> NormalizedDocument:
        """Parse a document file into a NormalizedDocument representation.

        Args:
            file_path: Absolute path to the document file.
            options: Parse options (OCR, tables, page limits).

        Returns:
            High-fidelity NormalizedDocument instance.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If file format is corrupt or unsupported.
            RuntimeError: If parsing fails unrecoverably.
        """
        ...
