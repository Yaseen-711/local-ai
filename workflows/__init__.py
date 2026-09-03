"""Local AI Foundation Workflow Layer.

Provides shared conventions and data contracts for workflow execution.
Workflows are plain Python classes that receive connectors via constructor injection.
"""

from workflows.text_analysis import (
    AnalysisDepth,
    AnalysisOptions,
    TextAnalysis,
    TextAnalysisWorkflow,
)
from workflows.types import WorkflowResult

__all__ = [
    "WorkflowResult",
    "AnalysisDepth",
    "AnalysisOptions",
    "TextAnalysis",
    "TextAnalysisWorkflow",
]
