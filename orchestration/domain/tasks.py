"""Task entity — a discrete, bounded unit of work in a plan.

A Task declares what needs to be done (via capability_id and parameters),
tracks its lifecycle state, and records execution attempts. It does NOT
execute work, resolve storage, or know which Python class implements the
capability.

Lifecycle transitions:
    PENDING → READY       (all dependencies satisfied)
    PENDING → BLOCKED     (upstream failed / blocked / cancelled)
    READY   → RUNNING     (attempt started)
    RUNNING → COMPLETED   (attempt succeeded)
    RUNNING → FAILED      (attempt failed; no retry in this phase)
    Any     → CANCELLED   (external cancellation)
    Any     → SKIPPED     (intentionally bypassed)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orchestration.domain.types import AttemptStatus, TaskStatus
from orchestration.domain.dependencies import Dependency
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.attempts import Attempt

# Terminal states — a task in one of these cannot transition further
# (except to CANCELLED, which is handled explicitly).
_TERMINAL_STATES = frozenset({
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.BLOCKED,
    TaskStatus.SKIPPED,
    TaskStatus.CANCELLED,
})


@dataclass
class Task:
    """A discrete, bounded unit of work within a plan.

    Attributes:
        task_id: Unique identifier for this task.
        plan_id: ID of the plan this task belongs to.
        title: Human-readable title.
        capability_id: Declarative identifier of the capability required
            (e.g. 'inference.chat', 'workflow.text_analysis'). Does NOT
            reference a Python class, framework, or execution backend.
        description: Optional detailed description.
        parameters: Small declarative task configuration.
        input_references: Logical references to input data produced
            elsewhere, keyed by logical name.
        dependencies: Execution dependency edges (upstream tasks that
            must complete before this task becomes READY).
        status: Current lifecycle state.
        created_at: When this task was defined.
        started_at: When the first attempt began (None until started).
        completed_at: When the task reached a terminal state.
        attempts: Ordered execution history.
        result: Final successful result (set on COMPLETED).
        error: Final error (set on FAILED).
    """
    task_id: str
    plan_id: str
    title: str
    capability_id: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_references: Dict[str, DataReference] = field(default_factory=dict)
    dependencies: List[Dependency] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: List[Attempt] = field(default_factory=list)
    result: Optional[TaskResult] = None
    error: Optional[TaskError] = None

    def start_attempt(self, attempt_id: str) -> Attempt:
        """Create and record a new execution attempt.

        Args:
            attempt_id: Unique identifier for the new attempt.

        Returns:
            The newly created Attempt in RUNNING status.

        Raises:
            ValueError: If the task is not in READY status, or if there
                is already a running attempt.
        """
        if self.status != TaskStatus.READY:
            raise ValueError(
                f"Cannot start attempt on task '{self.task_id}': "
                f"status is '{self.status.value}', expected 'ready'."
            )
        if any(a.status == AttemptStatus.RUNNING for a in self.attempts):
            raise ValueError(
                f"Task '{self.task_id}' already has a running attempt."
            )

        attempt = Attempt(
            attempt_id=attempt_id,
            task_id=self.task_id,
            attempt_number=len(self.attempts) + 1,
        )
        self.attempts.append(attempt)
        self.status = TaskStatus.RUNNING
        if self.started_at is None:
            self.started_at = attempt.started_at
        return attempt

    def complete_attempt(self, attempt_id: str, result: TaskResult) -> None:
        """Record a successful attempt and mark the task COMPLETED.

        Args:
            attempt_id: ID of the attempt to complete.
            result: Successful outcome.

        Raises:
            ValueError: If the task is not RUNNING or the attempt is
                not found or not in RUNNING status.
        """
        if self.status != TaskStatus.RUNNING:
            raise ValueError(
                f"Cannot complete attempt on task '{self.task_id}': "
                f"status is '{self.status.value}', expected 'running'."
            )
        attempt = self._get_attempt(attempt_id)
        attempt.mark_success(result)
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = attempt.completed_at

    def fail_attempt(self, attempt_id: str, error: TaskError) -> None:
        """Record a failed attempt and mark the task FAILED.

        No retry logic is implemented in this phase. A failed attempt
        immediately transitions the task to FAILED.

        Args:
            attempt_id: ID of the attempt to fail.
            error: Structured error describing the failure.

        Raises:
            ValueError: If the task is not RUNNING or the attempt is
                not found or not in RUNNING status.
        """
        if self.status != TaskStatus.RUNNING:
            raise ValueError(
                f"Cannot fail attempt on task '{self.task_id}': "
                f"status is '{self.status.value}', expected 'running'."
            )
        attempt = self._get_attempt(attempt_id)
        attempt.mark_failure(error)
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = attempt.completed_at

    def cancel(self) -> None:
        """Cancel this task.

        Raises:
            ValueError: If the task is already in a terminal state
                other than CANCELLED.
        """
        if self.status == TaskStatus.CANCELLED:
            return  # Idempotent
        if self.status in (TaskStatus.COMPLETED,):
            raise ValueError(
                f"Cannot cancel task '{self.task_id}': "
                f"already completed."
            )
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)

    def update_readiness(self, upstream_statuses: Dict[str, TaskStatus]) -> None:
        """Evaluate dependencies and transition PENDING → READY or BLOCKED.

        Simple success-dependency semantic: all upstream tasks must be
        COMPLETED for this task to become READY. If any upstream is
        FAILED, BLOCKED, or CANCELLED, this task becomes BLOCKED.

        Args:
            upstream_statuses: Mapping of upstream task IDs to their
                current statuses.
        """
        if self.status != TaskStatus.PENDING:
            return

        for dep in self.dependencies:
            upstream_status = upstream_statuses.get(dep.upstream_task_id)
            if upstream_status is None:
                return  # Upstream not yet known; stay PENDING
            if upstream_status in (TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED):
                self.status = TaskStatus.BLOCKED
                return
            if upstream_status != TaskStatus.COMPLETED:
                return  # Still waiting for upstream to finish

        # All dependencies satisfied
        self.status = TaskStatus.READY

    def _get_attempt(self, attempt_id: str) -> Attempt:
        """Look up an attempt by ID.

        Raises:
            ValueError: If no attempt with the given ID exists.
        """
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        raise ValueError(
            f"Attempt '{attempt_id}' not found on task '{self.task_id}'."
        )
