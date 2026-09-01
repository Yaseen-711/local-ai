"""Data structures defining model specifications, capabilities, and availability."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.common.types import ModelFormat, ModelRole


@dataclass(frozen=True)
class ModelCapabilities:
    """Declared functional capabilities of a model.
    
    These are declared in configuration rather than guessed from filenames or weights.
    """
    chat: bool = True
    code: bool = False
    reasoning: bool = False
    structured_output: bool = False
    context_window: int = 4096
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AvailabilityInfo:
    """Advisory filesystem availability state for a configured model.
    
    Note: Registry availability is advisory. Actual inference execution authority
    belongs strictly to the Provider and runtime.
    """
    is_available: bool
    resolved_path: Optional[Path] = None
    size_bytes: Optional[int] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class ModelDefinition:
    """Canonical model declaration loaded from configuration.
    
    Represents static model definitions known to the Foundation.
    Does NOT track active runtime loading state.
    """
    id: str
    display_name: str
    format: ModelFormat
    relative_path: Path
    supported_providers: List[str]
    aliases: List[str] = field(default_factory=list)
    roles: List[ModelRole] = field(default_factory=lambda: [ModelRole.GENERAL])
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_identifier(self, identifier: str) -> bool:
        """Check whether the given identifier matches the model's canonical ID or any alias."""
        return identifier == self.id or identifier in self.aliases
