"""Artifact generation capability package."""

from orchestration.capabilities.builtin.artifact.capability import (
    ArtifactGenerationCapability,
)
from orchestration.capabilities.builtin.artifact.generators import (
    DocxGenerator,
    PdfGenerator,
    PptxGenerator,
    XlsxGenerator,
)
from orchestration.capabilities.builtin.artifact.templates import (
    EngineeringCalculationSpec,
    ExecutivePresentationSpec,
    TechnicalApprovalNoteSpec,
    render_docx_approval_note,
    render_pptx_executive_presentation,
    render_template,
    render_xlsx_calculation_workbook,
)
from orchestration.capabilities.builtin.artifact.types import (
    ArtifactFormat,
    ArtifactGenerationRequest,
    ArtifactGenerationResponse,
)

__all__ = [
    "ArtifactFormat",
    "ArtifactGenerationCapability",
    "ArtifactGenerationRequest",
    "ArtifactGenerationResponse",
    "DocxGenerator",
    "EngineeringCalculationSpec",
    "ExecutivePresentationSpec",
    "PdfGenerator",
    "PptxGenerator",
    "TechnicalApprovalNoteSpec",
    "XlsxGenerator",
    "render_docx_approval_note",
    "render_pptx_executive_presentation",
    "render_template",
    "render_xlsx_calculation_workbook",
]

