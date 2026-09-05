"""Document normalization subsystem for Local AI Foundation RAG."""

from rag.normalization.interfaces import DocumentNormalizer
from rag.normalization.models import (
    NormalizedDocument,
    NormalizedElement,
    NormalizedElementType,
)
from rag.normalization.normalizer import (
    StandardDocumentNormalizer,
    clean_table_content,
    clean_text_whitespace,
)

__all__ = [
    "DocumentNormalizer",
    "StandardDocumentNormalizer",
    "NormalizedDocument",
    "NormalizedElement",
    "NormalizedElementType",
    "clean_text_whitespace",
    "clean_table_content",
]
