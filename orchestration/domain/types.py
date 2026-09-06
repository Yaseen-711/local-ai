"""Enumerations for orchestration domain lifecycle states and classifications."""

from enum import Enum


class GoalStatus(str, Enum):
    """Lifecycle states for a Goal."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanStatus(str, Enum):
    """Lifecycle states for a Plan."""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Lifecycle states for a Task."""
    PENDING = "pending"       # Waiting for dependencies to be satisfied
    READY = "ready"           # All dependencies satisfied; eligible for execution
    RUNNING = "running"       # An active attempt is in progress
    COMPLETED = "completed"   # Succeeded with a valid TaskResult
    FAILED = "failed"         # Attempt failed (no retry in this phase)
    BLOCKED = "blocked"       # An upstream dependency failed, was blocked, or was cancelled
    SKIPPED = "skipped"       # Intentionally bypassed
    CANCELLED = "cancelled"   # Terminated by external action


class AttemptStatus(str, Enum):
    """Outcome states for a single execution Attempt."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class TaskErrorCategory(str, Enum):
    """Classification of a task-level failure for structured error recording.

    These categories describe *what kind of failure* occurred, not where
    in the Python call stack the exception originated.
    """
    INFRASTRUCTURE = "infrastructure"  # Network, server unreachable, OOM
    TIMEOUT = "timeout"                # Deadline or time limit exceeded
    VALIDATION = "validation"          # Invalid input or malformed output
    CAPABILITY = "capability"          # Model/tool cannot satisfy the request
    EXECUTION = "execution"            # Failure inside workflow/capability logic
    CANCELLED = "cancelled"            # Aborted by external action
