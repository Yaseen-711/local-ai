"""Common schemas and RFC 7807 error envelopes for MRPL API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProblemDetail(BaseModel):
    """RFC 7807 problem detail response."""
    type: str = Field(default="about:blank", description="URI reference identifying the problem type")
    title: str = Field(description="Short human-readable summary of the problem type")
    status: int = Field(description="HTTP status code")
    detail: str = Field(description="Human-readable explanation specific to this occurrence")
    instance: Optional[str] = Field(default=None, description="URI reference identifying the specific occurrence")
    errors: Optional[List[Any]] = Field(default=None, description="Optional list of granular validation or cause details")


class DataReferenceSchema(BaseModel):
    """Logical reference to data or files generated or consumed in execution."""
    key: str
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactReferenceSchema(BaseModel):
    """External binary artifact produced by tasks (e.g. XLSX, DOCX, PDF)."""
    artifact_id: str
    name: str
    uri: str
    mime_type: str
    size_bytes: int
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
