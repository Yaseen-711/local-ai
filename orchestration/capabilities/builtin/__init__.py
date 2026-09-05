"""Built-in capabilities for Local AI Foundation."""

from orchestration.capabilities.builtin.inference import InferencePromptCapability
from orchestration.capabilities.builtin.workflow import TextAnalysisCapability

__all__ = [
    "InferencePromptCapability",
    "TextAnalysisCapability",
]
