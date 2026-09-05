"""Unit tests for TaskResult and TaskError value objects."""

import pytest

from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.types import TaskErrorCategory


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

def test_task_result_creation_defaults():
    """Verify TaskResult defaults to None output and empty collections."""
    result = TaskResult()
    assert result.output is None
    assert result.references == []
    assert result.artifacts == []
    assert result.metadata == {}


def test_task_result_with_output():
    """Verify TaskResult stores arbitrary output data."""
    result = TaskResult(output={"summary": "Financial report analysis"})
    assert result.output["summary"] == "Financial report analysis"


def test_task_result_with_references():
    """Verify TaskResult stores DataReference outputs."""
    ref = DataReference(key="analysis_output", source_task_id="task-1")
    result = TaskResult(references=[ref])
    assert len(result.references) == 1
    assert result.references[0].key == "analysis_output"


def test_task_result_with_artifacts():
    """Verify TaskResult stores ArtifactReference outputs."""
    art = ArtifactReference(
        artifact_id="art-1",
        name="report.pdf",
        uri="file:///output/report.pdf",
        mime_type="application/pdf",
        size_bytes=1024,
    )
    result = TaskResult(artifacts=[art])
    assert len(result.artifacts) == 1
    assert result.artifacts[0].name == "report.pdf"


def test_task_result_is_frozen():
    """Verify TaskResult is immutable."""
    result = TaskResult(output="data")
    with pytest.raises(AttributeError):
        result.output = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TaskError
# ---------------------------------------------------------------------------

def test_task_error_creation():
    """Verify TaskError stores structured failure information."""
    error = TaskError(
        message="Connection refused to llama-server",
        category=TaskErrorCategory.INFRASTRUCTURE,
        error_code="PROVIDER_UNAVAILABLE",
    )
    assert error.message == "Connection refused to llama-server"
    assert error.category == TaskErrorCategory.INFRASTRUCTURE
    assert error.error_code == "PROVIDER_UNAVAILABLE"
    assert error.details == {}
    assert error.cause_exception_type is None


def test_task_error_with_details():
    """Verify TaskError accepts arbitrary structured details."""
    error = TaskError(
        message="Validation failed",
        category=TaskErrorCategory.VALIDATION,
        details={"field": "summary", "reason": "expected str, got int"},
    )
    assert error.details["field"] == "summary"


def test_task_error_from_exception():
    """Verify TaskError.from_exception captures exception info."""
    try:
        raise ConnectionError("Server unreachable at 127.0.0.1:8080")
    except ConnectionError as exc:
        error = TaskError.from_exception(exc, category=TaskErrorCategory.INFRASTRUCTURE)

    assert "Server unreachable" in error.message
    assert error.category == TaskErrorCategory.INFRASTRUCTURE
    assert error.error_code == "ConnectionError"
    assert error.cause_exception_type == "builtins.ConnectionError"


def test_task_error_from_exception_default_category():
    """Verify TaskError.from_exception uses EXECUTION as default category."""
    error = TaskError.from_exception(RuntimeError("oops"))
    assert error.category == TaskErrorCategory.EXECUTION


def test_task_error_is_frozen():
    """Verify TaskError is immutable."""
    error = TaskError(message="err", category=TaskErrorCategory.EXECUTION)
    with pytest.raises(AttributeError):
        error.message = "modified"  # type: ignore[misc]
