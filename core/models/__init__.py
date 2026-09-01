"""Model definition and registry schemas for Local AI Foundation."""

from core.models.registry import ModelRegistry
from core.models.schema import (
    AvailabilityInfo,
    ModelCapabilities,
    ModelDefinition,
)

__all__ = [
    "AvailabilityInfo",
    "ModelCapabilities",
    "ModelDefinition",
    "ModelRegistry",
]
