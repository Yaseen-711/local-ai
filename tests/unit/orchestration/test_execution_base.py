"""Unit tests for PlanRunner protocol and execution lifecycle value objects."""

from datetime import datetime, timezone
import pytest

from orchestration.domain.plans import Plan
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.types import PlanStatus, TaskErrorCategory, TaskStatus
from orchestration.execution.base import (
    ExecutionHandle,
    PlanExecutionResult,
    PlanExecutionSnapshot,
    PlanRunner,
)


def test_execution_handle_creation():
    """Verify ExecutionHandle creation and immutability."""
    handle = ExecutionHandle(
        execution_id="exec-123",
        plan_id="plan-456",
        backend_info={"engine": "in_process"},
    )
    assert handle.execution_id == "exec-123"
    assert handle.plan_id == "plan-456"
    assert handle.backend_info["engine"] == "in_process"

    with pytest.raises(AttributeError):
        handle.execution_id = "other"  # type: ignore[misc]


def test_plan_execution_snapshot_creation():
    """Verify PlanExecutionSnapshot attributes and immutability."""
    snapshot = PlanExecutionSnapshot(
        execution_id="exec-1",
        plan_id="plan-1",
        status=PlanStatus.ACTIVE,
        task_statuses={"t1": TaskStatus.COMPLETED, "t2": TaskStatus.RUNNING},
        is_terminal=False,
    )
    assert snapshot.execution_id == "exec-1"
    assert snapshot.plan_id == "plan-1"
    assert snapshot.status == PlanStatus.ACTIVE
    assert snapshot.task_statuses["t1"] == TaskStatus.COMPLETED
    assert not snapshot.is_terminal

    with pytest.raises(AttributeError):
        snapshot.status = PlanStatus.COMPLETED  # type: ignore[misc]


def test_plan_execution_result_creation():
    """Verify PlanExecutionResult attributes and immutability."""
    now = datetime.now(timezone.utc)
    res = TaskResult(output="done")
    err = TaskError(message="failed", category=TaskErrorCategory.EXECUTION)

    result = PlanExecutionResult(
        execution_id="exec-1",
        plan_id="plan-1",
        status=PlanStatus.FAILED,
        task_results={"t1": res},
        task_errors={"t2": err},
        started_at=now,
        completed_at=now,
    )
    assert result.execution_id == "exec-1"
    assert result.plan_id == "plan-1"
    assert result.status == PlanStatus.FAILED
    assert result.task_results["t1"].output == "done"
    assert result.task_errors["t2"].message == "failed"
    assert result.started_at == now
    assert result.completed_at == now

    with pytest.raises(AttributeError):
        result.status = PlanStatus.COMPLETED  # type: ignore[misc]


def test_plan_runner_protocol_structural_subtyping():
    """Verify class satisfying all methods is recognized as PlanRunner."""

    class ConformingRunner:
        def start(self, plan: Plan) -> ExecutionHandle:
            return ExecutionHandle(execution_id="e1", plan_id=plan.plan_id)

        def wait(self, handle: ExecutionHandle, timeout=None) -> PlanExecutionResult:
            return PlanExecutionResult(
                execution_id=handle.execution_id,
                plan_id=handle.plan_id,
                status=PlanStatus.COMPLETED,
            )

        def get_status(self, handle: ExecutionHandle) -> PlanExecutionSnapshot:
            return PlanExecutionSnapshot(
                execution_id=handle.execution_id,
                plan_id=handle.plan_id,
                status=PlanStatus.COMPLETED,
                task_statuses={},
                is_terminal=True,
            )

        def cancel(self, handle: ExecutionHandle) -> None:
            pass

        def run(self, plan: Plan) -> Plan:
            return plan

    assert isinstance(ConformingRunner(), PlanRunner)


def test_plan_runner_protocol_incomplete_class_fails():
    """Verify class missing methods is not recognized as PlanRunner."""

    class IncompleteRunner:
        def run(self, plan: Plan) -> Plan:
            return plan

    assert not isinstance(IncompleteRunner(), PlanRunner)
