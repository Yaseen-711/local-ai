"""Inference connector capability abstraction and Foundation implementation.

This module provides the capability integration boundary between higher-level
orchestration/workflow layers and model inference infrastructure.
"""

from typing import Any, Optional, Protocol, runtime_checkable

from core.foundation import FoundationCore
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
)


@runtime_checkable
class InferenceConnector(Protocol):
    """Structural protocol defining the capability boundary for model inference.
    
    Workflows and orchestrators should depend on this protocol rather than
    concrete infrastructure containers, enabling dependency injection and testing.
    """

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Execute normalized model inference."""
        ...

    def infer_prompt(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        request_id: Optional[str] = None,
    ) -> InferenceResponse:
        """Execute single-turn prompt inference."""
        ...


class FoundationInferenceConnector:
    """In-process connector bridging callers to FoundationCore.
    
    Satisfies InferenceConnector protocol by delegating normalized inference
    execution directly to FoundationCore.
    """

    def __init__(self, core: FoundationCore) -> None:
        """Initialize connector with an active FoundationCore instance.
        
        Args:
            core: Configured FoundationCore instance.
        """
        self._core = core

    @property
    def core(self) -> FoundationCore:
        """Underlying FoundationCore instance."""
        return self._core

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Execute normalized model inference via FoundationCore.
        
        Args:
            request: Normalized inference request.
            
        Returns:
            Normalized inference response.
        """
        return self._core.infer(request)

    def infer_prompt(
        self,
        model_id: Optional[str] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InferenceResponse:
        """Execute single-turn prompt inference via FoundationCore.
        
        Args:
            model_id: Target model identifier or alias (defaults to 'default').
            prompt: User prompt content.
            system_prompt: Optional system instructions.
            options: Optional generation parameters.
            request_id: Optional tracking identifier.
            **kwargs: Flexible keyword arguments (temperature, max_tokens, etc.).
            
        Returns:
            Normalized inference response.
        """
        # Handle flexible calling conventions
        if model_id is not None and prompt is None and "prompt" not in kwargs:
            prompt = model_id
            target_model = "default"
        else:
            target_model = model_id or kwargs.pop("model_id", "default")

        target_prompt = prompt if prompt is not None else kwargs.pop("prompt", "")

        if options is None:
            temp = kwargs.pop("temperature", None)
            max_tok = kwargs.pop("max_tokens", None)
            if temp is not None or max_tok is not None:
                options = GenerationOptions(
                    temperature=float(temp) if temp is not None else 0.7,
                    max_tokens=int(max_tok) if max_tok is not None else 1024,
                )

        return self._core.infer_prompt(
            model_id=target_model,
            prompt=target_prompt,
            system_prompt=system_prompt,
            options=options,
            request_id=request_id,
        )
