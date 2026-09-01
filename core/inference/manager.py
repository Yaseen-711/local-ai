"""Provider Manager for Local AI Foundation.

Coordinates provider selection, inference routing, and thread-safe provider registration.
Does NOT duplicate static model definitions owned by ModelRegistry.
"""

import threading
from typing import Dict, List, Optional, Union

from core.common.errors import ProviderNotFoundError
from core.common.types import RuntimeState
from core.inference.provider import BaseProvider
from core.inference.types import InferenceRequest, InferenceResponse
from core.models.registry import ModelRegistry
from core.models.schema import ModelDefinition


class ProviderManager:
    """Manages registered inference providers and coordinates execution with ModelRegistry."""

    def __init__(self, registry: ModelRegistry) -> None:
        """Initialize the ProviderManager.
        
        Args:
            registry: Active ModelRegistry instance.
        """
        self._registry = registry
        self._providers: Dict[str, BaseProvider] = {}
        self._lock = threading.RLock()

    @property
    def registry(self) -> ModelRegistry:
        """Active ModelRegistry reference."""
        return self._registry

    def register_provider(self, provider: BaseProvider) -> None:
        """Register an inference provider instance.
        
        Thread-safe.
        """
        with self._lock:
            self._providers[provider.provider_name] = provider

    def unregister_provider(self, provider_name: str) -> None:
        """Unregister a provider by name.
        
        Thread-safe.
        """
        with self._lock:
            self._providers.pop(provider_name, None)

    def get_provider(self, provider_name: str) -> BaseProvider:
        """Retrieve a registered provider by name.
        
        Raises:
            ProviderNotFoundError: If the provider is not registered.
        """
        with self._lock:
            provider = self._providers.get(provider_name)
            if not provider:
                raise ProviderNotFoundError(
                    f"Provider '{provider_name}' is not registered. "
                    f"Available providers: {list(self._providers.keys())}"
                )
            return provider

    def list_providers(self) -> List[str]:
        """List names of all registered providers."""
        with self._lock:
            return list(self._providers.keys())

    def get_provider_for_model(self, model_def: ModelDefinition) -> BaseProvider:
        """Select the first available registered provider compatible with the model.
        
        Args:
            model_def: Target model definition.
            
        Returns:
            Compatible BaseProvider instance.
            
        Raises:
            ProviderNotFoundError: If no supported provider is registered.
        """
        with self._lock:
            for prov_name in model_def.supported_providers:
                if prov_name in self._providers:
                    return self._providers[prov_name]

            raise ProviderNotFoundError(
                f"No registered provider found for model '{model_def.id}'. "
                f"Model requires one of: {model_def.supported_providers}, "
                f"Registered providers: {list(self._providers.keys())}"
            )

    def get_runtime_state(self, provider_name: Optional[str] = None) -> Dict[str, RuntimeState]:
        """Query the runtime state of registered providers.
        
        Args:
            provider_name: Optional specific provider to query. If None, queries all.
            
        Returns:
            Dictionary mapping provider names to their current RuntimeState.
        """
        with self._lock:
            if provider_name:
                provider = self.get_provider(provider_name)
                return {provider.provider_name: provider.check_health()}

            return {name: prov.check_health() for name, prov in self._providers.items()}

    def execute_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Coordinate and execute an inference request end-to-end.
        
        1. Resolves model definition via ModelRegistry.
        2. Selects matching provider adapter.
        3. Executes inference and returns normalized response.
        
        Note: Registry disk availability is advisory; the provider and runtime
        remain the final authority on whether inference can execute.
        
        Args:
            request: Normalized InferenceRequest.
            
        Returns:
            Normalized InferenceResponse.
            
        Raises:
            ModelNotFoundError: If model_id is not in registry.
            ProviderNotFoundError: If no provider is registered for the model.
            ProviderUnavailableError: If runtime is offline.
            InferenceError: If runtime fails during execution.
        """
        # 1. Resolve model definition from registry
        model_def = self._registry.get_model(request.model_id)

        # 2. Select compatible provider
        provider = self.get_provider_for_model(model_def)

        # 3. Delegate execution to provider (authoritative runtime execution)
        return provider.infer(request, model_def)
