"""Schemas for synchronous direct capability execution."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from apps.api.schemas.common import ArtifactReferenceSchema, DataReferenceSchema


class DirectVisionRequest(BaseModel):
    """Input parameters for direct vision inspection."""
    file_id: Optional[str] = Field(default=None, description="Uploaded file ID from /api/v1/files/upload")
    file_path: Optional[str] = Field(default=None, description="Local path to image (within allowed repository/staging root)")
    query: Optional[str] = Field(default=None, description="Inspection instruction or prompt")
    model_id: Optional[str] = Field(default="qwen3.5-9b", description="Model ID for vision processing")
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)


class DirectDocumentRequest(BaseModel):
    """Input parameters for direct document understanding."""
    file_id: Optional[str] = Field(default=None, description="Uploaded file ID from /api/v1/files/upload")
    file_path: Optional[str] = Field(default=None, description="Local path to document")
    do_ocr: bool = Field(default=True, description="Enable OCR for scanned pages/images")
    extract_tables: bool = Field(default=True, description="Extract structured tables")
    extract_figures: bool = Field(default=False, description="Extract figures/diagrams")
    max_pages: Optional[int] = Field(default=None, ge=1)


class DirectArtifactRequest(BaseModel):
    """Input parameters for direct artifact generation."""
    artifact_type: str = Field(description="'xlsx', 'docx', or 'pdf'")
    filename: Optional[str] = Field(default=None, description="Target filename")
    title: Optional[str] = Field(default=None, description="Document/report title")
    data: Optional[Any] = Field(default=None, description="Tabular data (dict of sheets or table grid)")
    content: Optional[str] = Field(default=None, description="Markdown/prose content for docx/pdf")


class DirectTextAnalysisRequest(BaseModel):
    """Input parameters for direct text analysis."""
    text: str = Field(min_length=1, description="Raw text to analyze")
    depth: Optional[str] = Field(default="quick", description="'quick' or 'detailed'")
    focus: Optional[str] = Field(default=None, description="Specific analytical focus area")


class DirectCapabilityResponse(BaseModel):
    """Standardized response from synchronous direct capability execution."""
    capability_id: str
    status: str = "completed"
    output: Any = Field(description="Primary capability output payload")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    references: List[DataReferenceSchema] = Field(default_factory=list)
    artifacts: List[ArtifactReferenceSchema] = Field(default_factory=list)
