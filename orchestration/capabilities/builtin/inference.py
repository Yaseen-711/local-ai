"""Built-in inference prompt capability.

Bridges capability invocation to the InferenceConnector boundary.
Does NOT import or reference orchestration.domain.Task.
"""

from typing import Any, Dict, Optional

from connectors.inference import InferenceConnector
from core.inference.types import GenerationOptions
from orchestration.capabilities.base import CapabilityContext
from orchestration.domain.results import TaskResult


class InferencePromptCapability:
    """Capability executing single-turn prompt inference via InferenceConnector.

    Semantic contract:
        Required inputs/parameters:
            - 'prompt': str (non-empty prompt text)
        Optional parameters:
            - 'model_id': str (default: 'default')
            - 'system_prompt': Optional[str]
            - 'temperature': Optional[float]
            - 'top_p': Optional[float]
            - 'max_tokens': Optional[int]
            - 'seed': Optional[int]
    """

    def __init__(self, connector: InferenceConnector) -> None:
        """Initialize with an InferenceConnector implementation.

        Args:
            connector: Object satisfying InferenceConnector protocol.
        """
        self._connector = connector

    @property
    def capability_id(self) -> str:
        """Canonical identifier for prompt inference."""
        return "inference.prompt"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        """Execute single-turn inference.

        Args:
            parameters: Configuration parameters (model_id, temperature, etc.).
            inputs: Data inputs (prompt, etc.).
            context: Narrow invocation context.

        Returns:
            TaskResult with generated text output and usage metadata.

        Raises:
            ValueError: If 'prompt' is missing, empty, or not a string.
        """
        prompt = inputs.get("prompt") or parameters.get("prompt")
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Capability '{self.capability_id}' requires a non-empty string 'prompt' "
                f"in parameters or inputs."
            )

        model_id = str(parameters.get("model_id") or inputs.get("model_id") or "default")
        system_prompt = parameters.get("system_prompt") or inputs.get("system_prompt")
        if system_prompt is not None:
            system_prompt = str(system_prompt)

        # Build GenerationOptions if any generation hyperparameters are supplied
        options: Optional[GenerationOptions] = None
        temp = parameters.get("temperature")
        top_p = parameters.get("top_p")
        max_tokens = parameters.get("max_tokens")
        seed = parameters.get("seed")

        if any(v is not None for v in (temp, top_p, max_tokens, seed)):
            options = GenerationOptions(
                temperature=float(temp) if temp is not None else 0.7,
                top_p=float(top_p) if top_p is not None else 0.95,
                max_tokens=int(max_tokens) if max_tokens is not None else 512,
                seed=int(seed) if seed is not None else None,
            )

        response = self._connector.infer_prompt(
            model_id=model_id,
            prompt=prompt.strip(),
            system_prompt=system_prompt,
            options=options,
            request_id=context.execution_id,
        )

        return TaskResult(
            output=response.text,
            metadata={
                "model_id": response.model_id,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )
