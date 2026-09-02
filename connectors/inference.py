"""Inference connector capability abstraction and Foundation implementation.

This module provides the capability integration boundary between higher-level
orchestration/workflow layers and model inference infrastructure.
"""

from typing import Optional, Protocol, runtime_checkable

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
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        request_id: Optional[str] = None,
    ) -> InferenceResponse:
        """Execute single-turn prompt inference via FoundationCore.
        
        Args:
            model_id: Target model identifier or alias.
            prompt: User prompt content.
            system_prompt: Optional system instructions.
            options: Optional generation parameters.
            request_id: Optional tracking identifier.
            
        Returns:
            Normalized inference response.
        """
        return self._core.infer_prompt(
            model_id=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            options=options,
            request_id=request_id,
        )
