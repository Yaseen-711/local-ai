"""Built-in capabilities for Local AI Foundation."""

from orchestration.capabilities.builtin.agent import AgentCapability
from orchestration.capabilities.builtin.artifact import ArtifactGenerationCapability
from orchestration.capabilities.builtin.code import WorkspaceCodingCapability
from orchestration.capabilities.builtin.code_repair import CodeVerificationRepairCapability
from orchestration.capabilities.builtin.document import DocumentUnderstandingCapability
from orchestration.capabilities.builtin.inference import InferencePromptCapability
from orchestration.capabilities.builtin.vision import VisionInspectionCapability
from orchestration.capabilities.builtin.workflow import TextAnalysisCapability

__all__ = [
    "AgentCapability",
    "ArtifactGenerationCapability",
    "CodeVerificationRepairCapability",
    "DocumentUnderstandingCapability",
    "InferencePromptCapability",
    "TextAnalysisCapability",
    "VisionInspectionCapability",
    "WorkspaceCodingCapability",
]
