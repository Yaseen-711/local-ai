"""Data contracts for deterministic artifact generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ArtifactFormat(str, Enum):
    """Supported deterministic artifact formats."""
    XLSX = "xlsx"
    DOCX = "docx"
    PPTX = "pptx"
    PDF = "pdf"


@dataclass(frozen=True)
class ArtifactGenerationRequest:
    """Request specification for generating an artifact.

    Attributes:
        format: Target format (XLSX, DOCX, PDF).
        filename: Destination filename.
        title: Title of the document or workbook.
        data: Tabular or structured data (e.g. list of rows, list of dicts, or sheet dict).
        content: Text or markdown prose content for documents.
        metadata: Optional metadata to embed or record.
    """
    format: ArtifactFormat
    filename: str
    title: str = ""
    data: Optional[Union[List[List[Any]], List[Dict[str, Any]], Dict[str, Any]]] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactGenerationResponse:
    """Outcome of artifact generation.

    Attributes:
        artifact_id: Unique artifact identifier.
        name: Artifact human-readable name or filename.
        uri: Local file URI to the generated binary.
        size_bytes: Size of the artifact file in bytes.
        sha256: SHA-256 cryptographic digest of the file contents.
        mime_type: MIME type of the generated artifact.
    """
    artifact_id: str
    name: str
    uri: str
    size_bytes: int
    sha256: str
    mime_type: str
