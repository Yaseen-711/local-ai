"""Model Registry for Local AI Foundation.

The Model Registry owns:
- Declared model definitions from configuration
- Canonical model IDs and aliases
- Declared capabilities and metadata
- Resolved filesystem paths
- Cached advisory filesystem availability

The Model Registry does NOT own:
- Authoritative runtime loaded state
- Active provider process state
- Inference execution authority
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.common.errors import ModelNotFoundError, ModelRegistryError
from core.models.schema import AvailabilityInfo, ModelDefinition


class ModelRegistry:
    """Thread-safe catalog of known model definitions and advisory availability state."""

    def __init__(
        self,
        configs_dir: Union[str, Path],
        repo_root: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ) -> None:
        """Initialize the Model Registry.
        
        Args:
            configs_dir: Path to directory containing model TOML files.
            repo_root: Root path used to resolve relative model paths. Defaults to cwd.
            auto_load: Whether to load configurations and check availability on initialization.
        """
        self._configs_dir = Path(configs_dir).resolve()
        self._repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        self._lock = threading.RLock()

        self._models: Dict[str, ModelDefinition] = {}       # Keyed by canonical ID
        self._alias_map: Dict[str, str] = {}                # Map alias -> canonical ID
        self._availability: Dict[str, AvailabilityInfo] = {}  # Keyed by canonical ID

        if auto_load:
            self.refresh()

    @property
    def configs_dir(self) -> Path:
        """Directory containing model definitions."""
        return self._configs_dir

    @property
    def repo_root(self) -> Path:
        """Repository root used for path resolution."""
        return self._repo_root

    def refresh(self) -> None:
        """Re-scan configuration files and refresh advisory filesystem availability.
        
        Thread-safe. Does NOT perform per-request scans.
        """
        from core.config.loader import load_model_definitions_from_dir

        with self._lock:
            # 1. Load definitions from config files
            definitions = load_model_definitions_from_dir(self._configs_dir)

            new_models: Dict[str, ModelDefinition] = {}
            new_alias_map: Dict[str, str] = {}
            new_availability: Dict[str, AvailabilityInfo] = {}

            for model_id, model_def in definitions.items():
                new_models[model_id] = model_def
                new_alias_map[model_id] = model_id  # Canonical ID maps to itself

                for alias in model_def.aliases:
                    if alias in new_alias_map and new_alias_map[alias] != model_id:
                        existing_owner = new_alias_map[alias]
                        # Disambiguate or log collision
                        raise ModelRegistryError(
                            f"Alias collision: '{alias}' is defined by both "
                            f"'{existing_owner}' and '{model_id}'."
                        )
                    new_alias_map[alias] = model_id

                # 2. Check advisory filesystem availability
                avail_info = self._check_path_availability(model_def.relative_path)
                new_availability[model_id] = avail_info

            # Atomic swap under lock
            self._models = new_models
            self._alias_map = new_alias_map
            self._availability = new_availability

    def _check_path_availability(self, rel_or_abs_path: Path) -> AvailabilityInfo:
        """Check whether a model file exists on disk and record basic stat metadata."""
        target_path = rel_or_abs_path if rel_or_abs_path.is_absolute() else self._repo_root / rel_or_abs_path
        
        try:
            if target_path.is_file():
                size = target_path.stat().st_size
                return AvailabilityInfo(
                    is_available=True,
                    resolved_path=target_path,
                    size_bytes=size,
                    error_message=None,
                )
            else:
                return AvailabilityInfo(
                    is_available=False,
                    resolved_path=target_path,
                    size_bytes=None,
                    error_message=f"Model file does not exist at: {target_path}",
                )
        except Exception as e:
            return AvailabilityInfo(
                is_available=False,
                resolved_path=target_path,
                size_bytes=None,
                error_message=f"Error accessing model file at {target_path}: {e}",
            )

    def resolve_canonical_id(self, identifier: str) -> str:
        """Resolve a canonical ID or alias to its canonical model ID.
        
        Raises:
            ModelNotFoundError: If the identifier is unknown.
        """
        with self._lock:
            if identifier in self._alias_map:
                return self._alias_map[identifier]
            raise ModelNotFoundError(
                f"Model '{identifier}' is not configured in the registry. "
                f"Known models: {list(self._models.keys())}"
            )

    def get_model(self, identifier: str) -> ModelDefinition:
        """Get the ModelDefinition for a canonical ID or alias.
        
        Raises:
            ModelNotFoundError: If the model is not found in the registry.
        """
        canonical_id = self.resolve_canonical_id(identifier)
        with self._lock:
            return self._models[canonical_id]

    def get_availability(self, identifier: str) -> AvailabilityInfo:
        """Get cached advisory availability info for a model ID or alias.
        
        Note: This reflects the state at the time of the last refresh.
        
        Raises:
            ModelNotFoundError: If the model is not configured.
        """
        canonical_id = self.resolve_canonical_id(identifier)
        with self._lock:
            return self._availability.get(
                canonical_id,
                AvailabilityInfo(is_available=False, error_message="No availability recorded"),
            )

    def list_models(self) -> List[ModelDefinition]:
        """List all configured model definitions."""
        with self._lock:
            return list(self._models.values())

    def list_available_models(self) -> List[ModelDefinition]:
        """List all configured models whose files were confirmed available on disk at last refresh."""
        with self._lock:
            return [
                model_def
                for model_id, model_def in self._models.items()
                if self._availability.get(model_id, AvailabilityInfo(is_available=False)).is_available
            ]

    def is_known(self, identifier: str) -> bool:
        """Check if an identifier or alias is registered in the registry."""
        with self._lock:
            return identifier in self._alias_map
