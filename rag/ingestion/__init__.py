"""Document ingestion subsystem for Local AI Foundation RAG."""

from rag.ingestion.docling import (
    DoclingDocumentIngester,
    SUPPORTED_EXTENSIONS,
)
from rag.ingestion.errors import (
    DocumentParsingError,
    IngestionError,
    UnsupportedDocumentError,
)
from rag.ingestion.interfaces import DocumentIngester
from rag.ingestion.models import (
    DocumentElement,
    ElementType,
    IngestedDocument,
)

__all__ = [
    "DocumentIngester",
    "DoclingDocumentIngester",
    "IngestedDocument",
    "DocumentElement",
    "ElementType",
    "IngestionError",
    "UnsupportedDocumentError",
    "DocumentParsingError",
    "SUPPORTED_EXTENSIONS",
]
