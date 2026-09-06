"""Execution attempt record for task lifecycle history.

An Attempt is a discrete execution record of a single task execution.
It owns temporal information (started_at, completed_at), the outcome
status, and references to the immutable TaskResult or TaskError.

Attempts exist as execution history so that retry/recovery semantics
can be layered on in a future phase without redesigning the model.
In this phase, a task has at most one attempt.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orchestration.domain.types import AttemptStatus
from orchestration.domain.results import TaskError, TaskResult


@dataclass
class Attempt:
    """A discrete execution record of a task.

    Attributes:
        attempt_id: Unique identifier for this attempt.
        task_id: ID of the task this attempt belongs to.
        attempt_number: Sequential attempt number (1-based).
        started_at: When execution began.
        completed_at: When execution finished (None while running).
        status: Current outcome status.
        result: Successful outcome, if status is SUCCESS.
        error: Structured failure record, if status is FAILURE.
        metadata: Arbitrary execution telemetry or diagnostics.
    """
    attempt_id: str
    task_id: str
    attempt_number: int
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: AttemptStatus = AttemptStatus.RUNNING
    result: Optional[TaskResult] = None
    error: Optional[TaskError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_success(self, result: TaskResult) -> None:
        """Record a successful outcome for this attempt.

        Args:
            result: The successful task result.

        Raises:
            ValueError: If the attempt is not in RUNNING status.
        """
        if self.status != AttemptStatus.RUNNING:
            raise ValueError(
                f"Cannot mark attempt {self.attempt_id} as success: "
                f"current status is {self.status.value}, expected 'running'."
            )
        self.status = AttemptStatus.SUCCESS
        self.result = result
        self.completed_at = datetime.now(timezone.utc)

    def mark_failure(self, error: TaskError) -> None:
        """Record a failure outcome for this attempt.

        Args:
            error: Structured error describing the failure.

        Raises:
            ValueError: If the attempt is not in RUNNING status.
        """
        if self.status != AttemptStatus.RUNNING:
            raise ValueError(
                f"Cannot mark attempt {self.attempt_id} as failure: "
                f"current status is {self.status.value}, expected 'running'."
            )
        self.status = AttemptStatus.FAILURE
        self.error = error
        self.completed_at = datetime.now(timezone.utc)
