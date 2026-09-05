"""Built-in text analysis workflow capability.

Bridges capability invocation to TextAnalysisWorkflow.
Does NOT import or reference orchestration.domain.Task.
"""

from typing import Any, Dict, Optional

from orchestration.capabilities.base import CapabilityContext
from orchestration.domain.results import TaskResult
from workflows.text_analysis import AnalysisDepth, AnalysisOptions, TextAnalysisWorkflow


class TextAnalysisCapability:
    """Capability executing structured text analysis via TextAnalysisWorkflow.

    Semantic contract:
        Required inputs/parameters:
            - 'text': str (non-empty source text to analyze)
        Optional parameters:
            - 'depth': str ('quick' or 'detailed', default: 'quick')
            - 'focus': Optional[str] (analytical focus angle)
            - 'model_id': str (default: 'default')
            - 'temperature': Optional[float]
            - 'max_tokens': Optional[int]
    """

    def __init__(self, workflow: TextAnalysisWorkflow) -> None:
        """Initialize with a TextAnalysisWorkflow instance.

        Args:
            workflow: Configured TextAnalysisWorkflow instance.
        """
        self._workflow = workflow

    @property
    def capability_id(self) -> str:
        """Canonical identifier for text analysis."""
        return "workflow.text_analysis"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        """Execute text analysis workflow.

        Args:
            parameters: Configuration options (depth, focus, model_id, etc.).
            inputs: Data inputs (text, etc.).
            context: Narrow invocation context.

        Returns:
            TaskResult containing TextAnalysis output and workflow telemetry metadata.

        Raises:
            ValueError: If 'text' is missing, empty, or not a string.
        """
        text = inputs.get("text") or parameters.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Capability '{self.capability_id}' requires a non-empty string 'text' "
                f"in parameters or inputs."
            )

        depth_str = str(parameters.get("depth") or inputs.get("depth") or "quick").lower()
        try:
            depth = AnalysisDepth(depth_str)
        except ValueError:
            valid = [d.value for d in AnalysisDepth]
            raise ValueError(
                f"Invalid analysis depth '{depth_str}'. Expected one of: {valid}"
            )

        focus = parameters.get("focus") or inputs.get("focus")
        if focus is not None:
            focus = str(focus)

        model_id = str(parameters.get("model_id") or inputs.get("model_id") or "default")

        temp = parameters.get("temperature")
        max_tokens = parameters.get("max_tokens")

        options = AnalysisOptions(
            depth=depth,
            focus=focus,
            model_id=model_id,
            temperature=float(temp) if temp is not None else 0.2,
            max_tokens=int(max_tokens) if max_tokens is not None else 1024,
        )

        result = self._workflow.analyze(text=text.strip(), options=options)

        return TaskResult(
            output=result.output,
            metadata=result.metadata,
        )
