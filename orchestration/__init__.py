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
]
