"""Provider abstraction interface for Local AI Foundation."""

from abc import ABC, abstractmethod
from typing import Optional

from core.common.types import RuntimeState
from core.inference.types import InferenceRequest, InferenceResponse
from core.models.schema import ModelDefinition


class BaseProvider(ABC):
    """Abstract base class defining the contract for all inference providers/runtimes."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for this provider (e.g. 'llama_cpp')."""
        pass

    @abstractmethod
    def check_health(self) -> RuntimeState:
        """Query the runtime backend to assess its health and availability.
        
        Returns:
            RuntimeState: READY, UNAVAILABLE, or ERROR.
        """
        pass

    @abstractmethod
    def is_model_loaded(self, model_def: ModelDefinition) -> bool:
        """Check if the runtime is currently loaded/active for the given model.
        
        Returns:
            bool: True if runtime reports the model as loaded/serving.
        """
        pass

    @abstractmethod
    def infer(self, request: InferenceRequest, model_def: ModelDefinition) -> InferenceResponse:
        """Execute normalized inference against the provider runtime.
        
        Args:
            request: Normalized inference request.
            model_def: Target model definition from the registry.
            
        Returns:
            Normalized InferenceResponse.
            
        Raises:
            ProviderUnavailableError: If runtime backend cannot be reached.
            InferenceError: If runtime returns an error during inference.
            ProviderResponseError: If runtime returns a malformed response.
        """
        pass
