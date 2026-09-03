"""Local AI Foundation Core Package."""

from core.common.errors import (
    ConfigurationError,
    FoundationError,
    InferenceError,
    LifecycleConflictError,
    ModelNotFoundError,
    ModelRegistryError,
    ModelUnavailableError,
    ProviderError,
    ProviderNotFoundError,
    ProviderResponseError,
    ProviderUnavailableError,
    SyntaxParsingError,
    WorkflowError,
)
from core.common.types import (
    FinishReason,
    MessageRole,
    ModelFormat,
    ModelRole,
    RuntimeState,
)
from core.config.loader import (
    load_model_definition,
    load_model_definitions_from_dir,
    load_settings,
)
from core.config.settings import (
    FoundationSettings,
    LlamaCppProviderSettings,
    Settings,
)
from core.foundation import FoundationCore
from core.inference.manager import ProviderManager
from core.inference.provider import BaseProvider
from core.inference.providers.llama_cpp import LlamaCppProvider
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
    Message,
    OutputConstraint,
    TokenUsage,
)

from core.models.registry import ModelRegistry
from core.models.schema import (
    AvailabilityInfo,
    ModelCapabilities,
    ModelDefinition,
)

__all__ = [
    # High-level Foundation
    "FoundationCore",
    # Types & Enums
    "ModelFormat",
    "ModelRole",
    "RuntimeState",
    "MessageRole",
    "FinishReason",
    # Errors
    "FoundationError",
    "ConfigurationError",
    "ModelRegistryError",
    "ModelNotFoundError",
    "ModelUnavailableError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderUnavailableError",
    "ProviderResponseError",
    "InferenceError",
    "LifecycleConflictError",
    "SyntaxParsingError",
    "WorkflowError",
    # Config & Settings
    "load_model_definition",
    "load_model_definitions_from_dir",
    "load_settings",
    "FoundationSettings",
    "LlamaCppProviderSettings",
    "Settings",
    # Model Schema & Registry
    "ModelCapabilities",
    "AvailabilityInfo",
    "ModelDefinition",
    "ModelRegistry",
    # Inference & Providers
    "BaseProvider",
    "ProviderManager",
    "LlamaCppProvider",
    "Message",
    "GenerationOptions",
    "OutputConstraint",
    "InferenceRequest",
    "TokenUsage",
    "InferenceResponse",
]
