"""Domain exceptions for Local AI Foundation."""


class FoundationError(Exception):
    """Base class for all Local AI Foundation domain exceptions."""
    pass


class ConfigurationError(FoundationError):
    """Raised when configuration files are missing, unreadable, or syntactically/semantically invalid."""
    pass


class ModelRegistryError(FoundationError):
    """Base exception for model registry operations."""
    pass


class ModelNotFoundError(ModelRegistryError):
    """Raised when a requested model ID or alias is not configured in the registry."""
    pass


class ModelUnavailableError(ModelRegistryError):
    """Raised when a configured model file is not present on disk or is inaccessible."""
    pass


class ProviderError(FoundationError):
    """Base exception for provider and runtime failures."""
    pass


class ProviderNotFoundError(ProviderError):
    """Raised when no compatible provider is registered or available for a given model."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when the backend runtime (e.g. llama-server) is offline or unreachable."""
    pass


class ProviderResponseError(ProviderError):
    """Raised when the runtime returns a malformed or unexpected payload."""
    pass


class InferenceError(ProviderError):
    """Raised when inference execution fails at the runtime level."""
    pass


class LifecycleConflictError(ProviderError):
    """Raised when a concurrent or conflicting lifecycle operation is attempted on Core state."""
    pass


class WorkflowError(FoundationError):
    """Raised when a workflow-level operation fails.

    Distinct from infrastructure errors (ProviderError, ModelRegistryError).
    Workflow authors may raise this to signal domain-level failures,
    optionally wrapping an underlying infrastructure exception as the cause
    via standard Python exception chaining (raise WorkflowError(...) from cause).
    """
    pass
