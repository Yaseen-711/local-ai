"""File upload and staging schemas for MRPL API."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class FileUploadResponse(BaseModel):
    """Metadata returned upon successful file upload and staging."""
    file_id: str = Field(description="Unique opaque identifier for the staged file")
    filename: str = Field(description="Sanitized original filename")
    mime_type: str = Field(description="Detected or verified MIME type")
    size_bytes: int = Field(description="File size in bytes")
    sha256: str = Field(description="Cryptographic SHA-256 digest computed during streaming upload")
    uri: str = Field(description="Local file URI pointing to the sandboxed staged file")
    created_at: datetime = Field(description="Timestamp when the file was ingested")


class FileMetadataResponse(BaseModel):
    """Detailed file metadata."""
    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    uri: str
    exists: bool
    created_at: Optional[datetime] = None
