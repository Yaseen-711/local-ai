"""Structured outcome value objects for task attempts.

TaskResult and TaskError are immutable records that capture the outcome
of a single execution attempt. They are value objects — not runtime
exceptions. Python exceptions are converted into TaskError by the future
execution boundary; successful workflow/capability outputs are mapped
into TaskResult.

Temporal information (started_at, completed_at) belongs on the Attempt
that owns these objects, not on the result/error themselves.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from orchestration.domain.types import TaskErrorCategory
from orchestration.domain.references import ArtifactReference, DataReference


@dataclass(frozen=True)
class TaskResult:
    """Successful or completed output of a task attempt.

    Attributes:
        output: Primary output data (type depends on the capability).
        references: Logical output references produced by this task.
        artifacts: Large generated artifacts stored externally.
        metadata: Arbitrary diagnostic or downstream metadata.
    """
    output: Any = None
    references: List[DataReference] = field(default_factory=list)
    artifacts: List[ArtifactReference] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskError:
    """Structured, serializable failure record of a task attempt.

    This is a value object that *describes* a failure — it is NOT a
    Python exception. The future execution boundary converts runtime
    exceptions into TaskError instances.

    Attributes:
        message: Human-readable description of the failure.
        category: Classification of the failure kind.
        error_code: Machine-readable error identifier.
        details: Arbitrary structured details for debugging.
        cause_exception_type: Fully-qualified name of the original
            Python exception, if the error originated from one.
    """
    message: str
    category: TaskErrorCategory
    error_code: str = "TASK_EXECUTION_ERROR"
    details: Dict[str, Any] = field(default_factory=dict)
    cause_exception_type: Optional[str] = None

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        category: TaskErrorCategory = TaskErrorCategory.EXECUTION,
    ) -> "TaskError":
        """Convenience factory to create a TaskError from a Python exception.

        This is a mapping helper for future execution boundaries. It does
        not catch or re-raise exceptions.

        Args:
            exc: The source exception.
            category: Error classification.

        Returns:
            A frozen TaskError value object.
        """
        return cls(
            message=str(exc),
            category=category,
            error_code=type(exc).__name__,
            cause_exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
        )
