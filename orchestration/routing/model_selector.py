"""Model selection policy resolving abstract ModelTier to concrete model IDs."""

from typing import Dict, Optional

from core.models.registry import ModelRegistry
from orchestration.routing.types import ModelTier


DEFAULT_TIER_MAPPING: Dict[ModelTier, str] = {
    ModelTier.LIGHTWEIGHT: "qwen3.5-0.8b",
    ModelTier.REASONING: "qwen3.5-9b",
}


class ModelSelectionPolicy:
    """Policy mapping abstract ModelTier to concrete model IDs using ModelRegistry as catalog.

    Keeps model tiering as a decision/policy abstraction, preventing hardcoded
    model references in higher-level orchestration logic.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        tier_mapping: Optional[Dict[ModelTier, str]] = None,
    ) -> None:
        self._registry = registry
        self._tier_mapping: Dict[ModelTier, str] = (
            DEFAULT_TIER_MAPPING.copy() if tier_mapping is None else tier_mapping
        )

    def resolve_model_id(self, tier: ModelTier) -> str:
        """Resolve an abstract ModelTier to a concrete model identifier."""
        # 1. Check explicit policy tier mapping
        if tier in self._tier_mapping:
            candidate = self._tier_mapping[tier]
            if self._registry.is_known(candidate):
                return self._registry.get_model(candidate).id

        # 2. Check if default alias exists in registry
        if self._registry.is_known("default"):
            return self._registry.get_model("default").id

        # 3. Fall back to first available model in registry
        models = self._registry.list_models()
        if models:
            return models[0].id

        raise ValueError(f"No suitable model found in registry for tier '{tier.value}'.")
