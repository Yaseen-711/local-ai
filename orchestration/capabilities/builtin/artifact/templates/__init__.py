"""Industrial deliverable templates package."""

from orchestration.capabilities.builtin.artifact.templates.renderers import (
    SUPPORTED_TEMPLATES,
    render_docx_approval_note,
    render_pptx_executive_presentation,
    render_template,
    render_xlsx_calculation_workbook,
)
from orchestration.capabilities.builtin.artifact.templates.specs import (
    ApprovalSignOff,
    CalculationStep,
    EngineeringCalculationSpec,
    ExecutivePresentationSpec,
    InspectionTagItem,
    MetricCard,
    ParameterDefinition,
    PresentationSlideSpec,
    TechnicalApprovalNoteSpec,
    VerificationEvidence,
    create_demo_approval_note,
    create_demo_calculation_workbook,
    create_demo_executive_presentation,
)

__all__ = [
    "ApprovalSignOff",
    "CalculationStep",
    "EngineeringCalculationSpec",
    "ExecutivePresentationSpec",
    "InspectionTagItem",
    "MetricCard",
    "ParameterDefinition",
    "PresentationSlideSpec",
    "SUPPORTED_TEMPLATES",
    "TechnicalApprovalNoteSpec",
    "VerificationEvidence",
    "create_demo_approval_note",
    "create_demo_calculation_workbook",
    "create_demo_executive_presentation",
    "render_docx_approval_note",
    "render_pptx_executive_presentation",
    "render_template",
    "render_xlsx_calculation_workbook",
]
