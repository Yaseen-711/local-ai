"""Pure domain models for the RAG subsystem.

These dataclasses define the core data contracts for documents, chunks, and retrieval
results. They have zero dependencies on external frameworks, databases, or vector stores.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Document:
    """Represents an ingested source document."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Represents a chunk extracted from a parent document for indexing and retrieval."""

    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """Represents a chunk match returned by vector search or retrieval."""

    chunk: Chunk
    score: float
