"""Orchestration domain exceptions.

These exceptions are for orchestration-layer errors that are distinct
from core infrastructure errors (ProviderError, ModelRegistryError)
and workflow errors (WorkflowError). They exist at the orchestration
layer in the dependency hierarchy.
"""


class OrchestrationError(Exception):
    """Base class for all orchestration domain exceptions."""
    pass


class PlanValidationError(OrchestrationError):
    """Raised when a plan fails structural validation (e.g. DAG cycle)."""
    pass


class TaskLifecycleError(OrchestrationError):
    """Raised when an invalid task lifecycle transition is attempted."""
    pass


class CapabilityError(OrchestrationError):
    """Base exception for capability-related errors."""
    pass


class CapabilityNotFoundError(CapabilityError):
    """Raised when a requested capability_id is not found in the registry."""
    pass


class CapabilityRegistryError(CapabilityError):
    """Raised when a capability cannot be registered (e.g. duplicate ID)."""
    pass


class CapabilityUnavailableError(CapabilityError):
    """Raised when a requested capability is unavailable in the runtime environment."""
    pass
