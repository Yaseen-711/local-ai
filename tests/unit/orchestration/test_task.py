"""Unit tests for Task lifecycle, readiness, and attempt management."""

import pytest

from orchestration.domain.attempts import Attempt
from orchestration.domain.dependencies import Dependency
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import AttemptStatus, TaskErrorCategory, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id: str = "task-1", deps: list = None, **kwargs) -> Task:
    """Create a minimal Task for testing."""
    defaults = dict(
        plan_id="plan-1",
        title=f"Task {task_id}",
        capability_id="test.capability",
        dependencies=deps or [],
    )
    defaults.update(kwargs)
    return Task(task_id=task_id, **defaults)


# ---------------------------------------------------------------------------
# Task Creation
# ---------------------------------------------------------------------------

def test_task_creation_defaults():
    """Verify Task is created with PENDING status and sensible defaults."""
    task = _make_task()
    assert task.task_id == "task-1"
    assert task.status == TaskStatus.PENDING
    assert task.capability_id == "test.capability"
    assert task.parameters == {}
    assert task.input_references == {}
    assert task.dependencies == []
    assert task.attempts == []
    assert task.result is None
    assert task.error is None
    assert task.started_at is None
    assert task.completed_at is None


def test_task_with_parameters_and_references():
    """Verify Task accepts parameters and input references."""
    ref = DataReference(key="source_text", source_task_id="upstream-1")
    task = _make_task(
        parameters={"model_id": "qwen3.5-9b", "temperature": 0.3},
        input_references={"source": ref},
    )
    assert task.parameters["model_id"] == "qwen3.5-9b"
    assert task.input_references["source"].source_task_id == "upstream-1"


# ---------------------------------------------------------------------------
# Attempt Lifecycle
# ---------------------------------------------------------------------------

def test_task_start_attempt_from_ready():
    """Verify starting an attempt transitions READY → RUNNING."""
    task = _make_task()
    task.status = TaskStatus.READY  # Simulating dependency resolution
    attempt = task.start_attempt("attempt-1")
    assert task.status == TaskStatus.RUNNING
    assert attempt.attempt_id == "attempt-1"
    assert attempt.task_id == "task-1"
    assert attempt.attempt_number == 1
    assert attempt.status == AttemptStatus.RUNNING
    assert task.started_at is not None
    assert len(task.attempts) == 1


def test_task_start_attempt_not_ready_raises():
    """Verify starting an attempt from PENDING raises ValueError."""
    task = _make_task()
    with pytest.raises(ValueError, match="expected 'ready'"):
        task.start_attempt("attempt-1")


def test_task_start_attempt_already_running_raises():
    """Verify starting a second concurrent attempt raises ValueError."""
    task = _make_task()
    task.status = TaskStatus.READY
    task.start_attempt("attempt-1")
    # Task is now RUNNING — force back to READY to test the concurrent check
    # But the real guard is on having a running attempt, regardless of task status
    task.status = TaskStatus.READY  # Artificially reset for test
    with pytest.raises(ValueError, match="already has a running attempt"):
        task.start_attempt("attempt-2")


def test_task_complete_attempt():
    """Verify successful attempt transitions RUNNING → COMPLETED."""
    task = _make_task()
    task.status = TaskStatus.READY
    task.start_attempt("attempt-1")

    result = TaskResult(output={"summary": "Done"})
    task.complete_attempt("attempt-1", result)

    assert task.status == TaskStatus.COMPLETED
    assert task.result is result
    assert task.completed_at is not None
    assert task.attempts[0].status == AttemptStatus.SUCCESS
    assert task.attempts[0].result is result


def test_task_complete_attempt_not_running_raises():
    """Verify completing when task is not RUNNING raises ValueError."""
    task = _make_task()
    task.status = TaskStatus.READY
    with pytest.raises(ValueError, match="expected 'running'"):
        task.complete_attempt("attempt-1", TaskResult())


def test_task_complete_attempt_unknown_id_raises():
    """Verify completing with a nonexistent attempt ID raises ValueError."""
    task = _make_task()
    task.status = TaskStatus.READY
    task.start_attempt("attempt-1")
    with pytest.raises(ValueError, match="not found"):
        task.complete_attempt("nonexistent", TaskResult())


def test_task_fail_attempt():
    """Verify failed attempt transitions RUNNING → FAILED."""
    task = _make_task()
    task.status = TaskStatus.READY
    task.start_attempt("attempt-1")

    error = TaskError(
        message="Model returned empty output",
        category=TaskErrorCategory.EXECUTION,
    )
    task.fail_attempt("attempt-1", error)

    assert task.status == TaskStatus.FAILED
    assert task.error is error
    assert task.completed_at is not None
    assert task.attempts[0].status == AttemptStatus.FAILURE
    assert task.attempts[0].error is error


def test_task_fail_attempt_not_running_raises():
    """Verify failing when task is not RUNNING raises ValueError."""
    task = _make_task()
    with pytest.raises(ValueError, match="expected 'running'"):
        task.fail_attempt("attempt-1", TaskError(
            message="err", category=TaskErrorCategory.EXECUTION,
        ))


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_task_cancel_from_pending():
    """Verify cancellation from PENDING."""
    task = _make_task()
    task.cancel()
    assert task.status == TaskStatus.CANCELLED
    assert task.completed_at is not None


def test_task_cancel_from_ready():
    """Verify cancellation from READY."""
    task = _make_task()
    task.status = TaskStatus.READY
    task.cancel()
    assert task.status == TaskStatus.CANCELLED


def test_task_cancel_from_running():
    """Verify cancellation from RUNNING."""
    task = _make_task()
    task.status = TaskStatus.READY
    task.start_attempt("attempt-1")
    task.cancel()
    assert task.status == TaskStatus.CANCELLED


def test_task_cancel_completed_raises():
    """Verify cancelling a COMPLETED task raises ValueError."""
    task = _make_task()
    task.status = TaskStatus.READY
    task.start_attempt("attempt-1")
    task.complete_attempt("attempt-1", TaskResult())
    with pytest.raises(ValueError, match="already completed"):
        task.cancel()


def test_task_cancel_idempotent():
    """Verify cancelling an already-cancelled task is a no-op."""
    task = _make_task()
    task.cancel()
    task.cancel()  # Should not raise
    assert task.status == TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# Readiness / Dependency Resolution
# ---------------------------------------------------------------------------

def test_task_readiness_no_dependencies():
    """Verify task with no dependencies transitions PENDING → READY."""
    task = _make_task()
    task.update_readiness({})
    assert task.status == TaskStatus.READY


def test_task_readiness_upstream_completed():
    """Verify task becomes READY when all upstreams are COMPLETED."""
    task = _make_task(deps=[
        Dependency(upstream_task_id="upstream-1", downstream_task_id="task-1"),
    ])
    task.update_readiness({"upstream-1": TaskStatus.COMPLETED})
    assert task.status == TaskStatus.READY


def test_task_readiness_upstream_still_running():
    """Verify task stays PENDING when upstream is still RUNNING."""
    task = _make_task(deps=[
        Dependency(upstream_task_id="upstream-1", downstream_task_id="task-1"),
    ])
    task.update_readiness({"upstream-1": TaskStatus.RUNNING})
    assert task.status == TaskStatus.PENDING


def test_task_readiness_upstream_failed_blocks():
    """Verify task becomes BLOCKED when upstream FAILED."""
    task = _make_task(deps=[
        Dependency(upstream_task_id="upstream-1", downstream_task_id="task-1"),
    ])
    task.update_readiness({"upstream-1": TaskStatus.FAILED})
    assert task.status == TaskStatus.BLOCKED


def test_task_readiness_upstream_cancelled_blocks():
    """Verify task becomes BLOCKED when upstream CANCELLED."""
    task = _make_task(deps=[
        Dependency(upstream_task_id="upstream-1", downstream_task_id="task-1"),
    ])
    task.update_readiness({"upstream-1": TaskStatus.CANCELLED})
    assert task.status == TaskStatus.BLOCKED


def test_task_readiness_upstream_blocked_blocks():
    """Verify task becomes BLOCKED when upstream is BLOCKED."""
    task = _make_task(deps=[
        Dependency(upstream_task_id="upstream-1", downstream_task_id="task-1"),
    ])
    task.update_readiness({"upstream-1": TaskStatus.BLOCKED})
    assert task.status == TaskStatus.BLOCKED


def test_task_readiness_multiple_deps_partial():
    """Verify task stays PENDING when only some upstreams are COMPLETED."""
    task = _make_task(deps=[
        Dependency(upstream_task_id="a", downstream_task_id="task-1"),
        Dependency(upstream_task_id="b", downstream_task_id="task-1"),
    ])
    task.update_readiness({"a": TaskStatus.COMPLETED, "b": TaskStatus.RUNNING})
    assert task.status == TaskStatus.PENDING


def test_task_readiness_multiple_deps_all_completed():
    """Verify task becomes READY when all upstreams are COMPLETED."""
    task = _make_task(deps=[
        Dependency(upstream_task_id="a", downstream_task_id="task-1"),
        Dependency(upstream_task_id="b", downstream_task_id="task-1"),
    ])
    task.update_readiness({"a": TaskStatus.COMPLETED, "b": TaskStatus.COMPLETED})
    assert task.status == TaskStatus.READY


def test_task_readiness_unknown_upstream_stays_pending():
    """Verify task stays PENDING when upstream status is unknown."""
    task = _make_task(deps=[
        Dependency(upstream_task_id="unknown", downstream_task_id="task-1"),
    ])
    task.update_readiness({})
    assert task.status == TaskStatus.PENDING


def test_task_readiness_skipped_from_non_pending():
    """Verify update_readiness is a no-op when task is not PENDING."""
    task = _make_task()
    task.status = TaskStatus.READY
    task.update_readiness({"upstream-1": TaskStatus.FAILED})
    # Should not change from READY to BLOCKED since it's not PENDING
    assert task.status == TaskStatus.READY
