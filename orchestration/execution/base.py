"""Execution boundary protocol and value objects for orchestration plan runners.

Defines the pluggable execution boundary between GoalOrchestrator and
concrete execution backends (InProcessPlanRunner for in-memory execution,
or TemporalPlanRunner for durable distributed execution).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from orchestration.domain.plans import Plan
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.types import PlanStatus, TaskStatus


@dataclass(frozen=True)
class ExecutionHandle:
    """Opaque durable reference to an in-flight or completed plan execution.

    Attributes:
        execution_id: Unique identifier for this execution run.
        plan_id: ID of the Plan being executed.
        backend_info: Optional runner-specific metadata (e.g. workflow_id, run_id).
    """

    execution_id: str
    plan_id: str
    backend_info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanExecutionSnapshot:
    """Non-blocking point-in-time snapshot of an active or completed plan execution.

    Attributes:
        execution_id: Unique identifier of the execution run.
        plan_id: ID of the Plan being executed.
        status: Current lifecycle state of the execution.
        task_statuses: Mapping of task IDs to their current TaskStatus.
        is_terminal: True if the execution has reached a final state.
    """

    execution_id: str
    plan_id: str
    status: PlanStatus
    task_statuses: Dict[str, TaskStatus]
    is_terminal: bool


@dataclass(frozen=True)
class PlanExecutionResult:
    """Terminal execution facts returned when a plan execution completes.

    Attributes:
        execution_id: Unique identifier of the completed execution.
        plan_id: ID of the Plan that was executed.
        status: Terminal PlanStatus (COMPLETED, FAILED, or CANCELLED).
        task_results: Results of successfully completed tasks keyed by task_id.
        task_errors: Errors of failed tasks keyed by task_id.
        started_at: Optional timestamp when execution started.
        completed_at: Optional timestamp when execution finished.
    """

    execution_id: str
    plan_id: str
    status: PlanStatus
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    task_errors: Dict[str, TaskError] = field(default_factory=dict)
    task_attempts: Dict[str, str] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@runtime_checkable
class PlanRunner(Protocol):
    """Execution boundary protocol for driving an orchestration Plan to completion.

    Supports durable execution by decoupling start, wait, status inspection,
    and cancellation. run() is provided as a convenience helper that combines
    start and wait.
    """

    def start(self, plan: Plan) -> ExecutionHandle:
        """Initiate plan execution asynchronously or non-blockingly.

        Args:
            plan: The Plan to execute.

        Returns:
            ExecutionHandle identifying the running execution.
        """
        ...

    def wait(
        self,
        handle: ExecutionHandle,
        timeout: Optional[float] = None,
    ) -> PlanExecutionResult:
        """Wait for an active execution to reach a terminal state.

        Args:
            handle: The ExecutionHandle returned by start().
            timeout: Optional timeout in seconds to wait before raising TimeoutError.

        Returns:
            PlanExecutionResult containing terminal execution facts.
        """
        ...

    def get_status(self, handle: ExecutionHandle) -> PlanExecutionSnapshot:
        """Query the non-blocking status of an execution.

        Args:
            handle: The ExecutionHandle to query.

        Returns:
            PlanExecutionSnapshot reflecting the latest progress.
        """
        ...

    def cancel(self, handle: ExecutionHandle) -> None:
        """Request graceful cancellation of a running execution.

        Args:
            handle: The ExecutionHandle to cancel.
        """
        ...

    def run(self, plan: Plan) -> Any:
        """Convenience method: start execution and wait for completion.

        Args:
            plan: The Plan to execute.

        Returns:
            Terminal execution outcome (PlanExecutionResult or executed Plan).
        """
        ...
