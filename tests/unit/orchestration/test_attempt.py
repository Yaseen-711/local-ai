"""Unit tests for Attempt lifecycle and transition guards."""

import pytest

from orchestration.domain.attempts import Attempt
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.types import AttemptStatus, TaskErrorCategory


def test_attempt_creation_defaults():
    """Verify Attempt is created in RUNNING status with sensible defaults."""
    attempt = Attempt(
        attempt_id="attempt-1",
        task_id="task-1",
        attempt_number=1,
    )
    assert attempt.attempt_id == "attempt-1"
    assert attempt.task_id == "task-1"
    assert attempt.attempt_number == 1
    assert attempt.status == AttemptStatus.RUNNING
    assert attempt.started_at is not None
    assert attempt.completed_at is None
    assert attempt.result is None
    assert attempt.error is None
    assert attempt.metadata == {}


def test_attempt_mark_success():
    """Verify RUNNING → SUCCESS transition."""
    attempt = Attempt(attempt_id="a-1", task_id="t-1", attempt_number=1)
    result = TaskResult(output="analysis complete")
    attempt.mark_success(result)

    assert attempt.status == AttemptStatus.SUCCESS
    assert attempt.result is result
    assert attempt.completed_at is not None
    assert attempt.error is None


def test_attempt_mark_success_not_running_raises():
    """Verify marking success from non-RUNNING raises ValueError."""
    attempt = Attempt(attempt_id="a-1", task_id="t-1", attempt_number=1)
    attempt.mark_success(TaskResult())
    with pytest.raises(ValueError, match="expected 'running'"):
        attempt.mark_success(TaskResult())


def test_attempt_mark_failure():
    """Verify RUNNING → FAILURE transition."""
    attempt = Attempt(attempt_id="a-1", task_id="t-1", attempt_number=1)
    error = TaskError(message="timeout", category=TaskErrorCategory.TIMEOUT)
    attempt.mark_failure(error)

    assert attempt.status == AttemptStatus.FAILURE
    assert attempt.error is error
    assert attempt.completed_at is not None
    assert attempt.result is None


def test_attempt_mark_failure_not_running_raises():
    """Verify marking failure from non-RUNNING raises ValueError."""
    attempt = Attempt(attempt_id="a-1", task_id="t-1", attempt_number=1)
    error = TaskError(message="err", category=TaskErrorCategory.EXECUTION)
    attempt.mark_failure(error)
    with pytest.raises(ValueError, match="expected 'running'"):
        attempt.mark_failure(error)


def test_attempt_cannot_succeed_after_failure():
    """Verify cannot mark success after already marking failure."""
    attempt = Attempt(attempt_id="a-1", task_id="t-1", attempt_number=1)
    attempt.mark_failure(TaskError(message="err", category=TaskErrorCategory.EXECUTION))
    with pytest.raises(ValueError, match="expected 'running'"):
        attempt.mark_success(TaskResult())


def test_attempt_metadata():
    """Verify arbitrary metadata is preserved."""
    attempt = Attempt(
        attempt_id="a-1",
        task_id="t-1",
        attempt_number=1,
        metadata={"latency_ms": 142.5, "provider": "llama_cpp"},
    )
    assert attempt.metadata["latency_ms"] == 142.5
    assert attempt.metadata["provider"] == "llama_cpp"
