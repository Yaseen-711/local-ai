"""Orchestration Domain Layer for Local AI Foundation.

Provides semantic domain contracts for goal-driven task orchestration:
Goal → Plan → PlanRevision → Task → Dependency → Attempt → TaskResult / TaskError.

These are pure domain entities and value objects. They manage lifecycle state,
relationships, validation, and readiness — but do NOT execute work, resolve
storage, or couple to any execution framework (Celery, Docker, etc.).

Dependency direction:
    apps → orchestration → workflows / connectors → core
"""

from orchestration.domain.types import (
    AttemptStatus,
    GoalStatus,
    PlanStatus,
    TaskErrorCategory,
    TaskStatus,
)
from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.dependencies import Dependency
from orchestration.domain.attempts import Attempt
from orchestration.domain.tasks import Task
from orchestration.domain.plans import Plan, PlanRevision
from orchestration.domain.goals import Goal
from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.execution.base import (
    ExecutionHandle,
    PlanExecutionResult,
    PlanExecutionSnapshot,
    PlanRunner,
)
from orchestration.execution.runner import InProcessPlanRunner
from orchestration.orchestrator import DirectGoalResult, GoalOrchestrator
from orchestration.persistence.base import (
    GoalRepository,
    OrchestrationRepository,
    PlanRepository,
)
from orchestration.persistence.repository import (
    PostgresGoalRepository,
    PostgresOrchestrationRepository,
    PostgresPlanRepository,
)
from orchestration.errors import (
    CapabilityError,
    CapabilityNotFoundError,
    CapabilityRegistryError,
    OrchestrationError,
    PlanValidationError,
    TaskLifecycleError,
)

__all__ = [
    # Enums
    "GoalStatus",
    "PlanStatus",
    "TaskStatus",
    "AttemptStatus",
    "TaskErrorCategory",
    # Value objects
    "DataReference",
    "ArtifactReference",
    "TaskResult",
    "TaskError",
    "Dependency",
    # Lifecycle entities
    "Attempt",
    "Task",
    "Plan",
    "PlanRevision",
    "Goal",
    # Capabilities
    "Capability",
    "CapabilityContext",
    "CapabilityRegistry",
    # Execution
    "ExecutionHandle",
    "PlanExecutionResult",
    "PlanExecutionSnapshot",
    "PlanRunner",
    "InProcessPlanRunner",
    # Orchestrator
    "DirectGoalResult",
    "GoalOrchestrator",
    # Persistence
    "GoalRepository",
    "PlanRepository",
    "OrchestrationRepository",
    "PostgresGoalRepository",
    "PostgresPlanRepository",
    "PostgresOrchestrationRepository",
    # Errors
    "OrchestrationError",
    "PlanValidationError",
    "TaskLifecycleError",
    "CapabilityError",
    "CapabilityNotFoundError",
    "CapabilityRegistryError",
]
