"""Exceptions for the RAG document ingestion layer."""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for document ingestion failures."""
    pass


class UnsupportedDocumentError(IngestionError):
    """Raised when an unsupported file format or extension is provided."""
    pass


class DocumentParsingError(IngestionError):
    """Raised when Docling or the parser fails to parse a document."""
    pass
