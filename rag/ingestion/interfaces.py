"""Interfaces and protocols for document ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union, runtime_checkable

from rag.ingestion.models import IngestedDocument


@runtime_checkable
class DocumentIngester(Protocol):
    """Protocol for components that ingest and extract structured content from files."""

    def ingest(self, file_path: Union[str, Path]) -> IngestedDocument:
        """Parse a local document and produce a structured IngestedDocument representation.

        Args:
            file_path: Path to the local file to ingest.

        Returns:
            Structured IngestedDocument with extracted elements and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedDocumentError: If the file extension/format is not supported.
            DocumentParsingError: If extraction fails during parsing.
        """
        ...

    def supports_format(self, file_path: Union[str, Path]) -> bool:
        """Check whether the given file path has a supported extension/format.

        Args:
            file_path: Path to the file.

        Returns:
            True if supported, False otherwise.
        """
        ...
