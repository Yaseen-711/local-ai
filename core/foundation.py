"""High-level Foundation entry point and factory."""

from pathlib import Path
from typing import Optional, Union

from core.config.loader import load_settings
from core.config.settings import Settings
from core.inference.manager import ProviderManager
from core.inference.providers.llama_cpp import LlamaCppProvider
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
)
from core.models.registry import ModelRegistry


class FoundationCore:
    """Convenience coordinator managing ModelRegistry and ProviderManager."""

    def __init__(
        self,
        registry: ModelRegistry,
        provider_manager: ProviderManager,
        settings: Settings,
    ) -> None:
        self._registry = registry
        self._provider_manager = provider_manager
        self._settings = settings

    @property
    def registry(self) -> ModelRegistry:
        """Active ModelRegistry."""
        return self._registry

    @property
    def provider_manager(self) -> ProviderManager:
        """Active ProviderManager."""
        return self._provider_manager

    @property
    def settings(self) -> Settings:
        """Active Settings."""
        return self._settings

    @property
    def repo_root(self) -> Path:
        """Repository root path used for resolving relative paths."""
        return self._registry.repo_root


    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Execute normalized inference.
        
        This is the formal reusable public execution boundary for higher-level workflows.
        Workflows may call it any number of times, in loops, or between pipeline steps.
        """
        return self._provider_manager.execute_inference(request)

    def infer_prompt(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        request_id: Optional[str] = None,
    ) -> InferenceResponse:
        """Convenience single-turn execution method.
        
        Allows workflows to execute inference without manually constructing an InferenceRequest.
        Internally delegates to self.infer() using the normalized contract.
        """
        request = InferenceRequest.from_prompt(
            model_id=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            options=options,
            request_id=request_id,
        )
        return self.infer(request)

    @classmethod
    def create(
        cls,
        repo_root: Optional[Union[str, Path]] = None,
        configs_dir: Optional[Union[str, Path]] = None,
        settings_path: Optional[Union[str, Path]] = None,
    ) -> "FoundationCore":
        """Factory method to initialize a FoundationCore instance with default settings.
        
        Args:
            repo_root: Repository root path (defaults to cwd).
            configs_dir: Directory containing model TOML files.
            settings_path: Path to settings.toml.
            
        Returns:
            Configured FoundationCore instance.
        """
        root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        settings_file = Path(settings_path).resolve() if settings_path else root / "configs" / "settings.toml"
        settings = load_settings(settings_file)

        cfg_dir = Path(configs_dir).resolve() if configs_dir else root / settings.foundation.configs_dir

        registry = ModelRegistry(configs_dir=cfg_dir, repo_root=root, auto_load=True)
        manager = ProviderManager(registry=registry)

        # Register default llama.cpp provider
        llama_provider = LlamaCppProvider(
            base_url=settings.llama_cpp.base_url,
            timeout_seconds=settings.llama_cpp.timeout_seconds,
        )
        manager.register_provider(llama_provider)

        return cls(
            registry=registry,
            provider_manager=manager,
            settings=settings,
        )
