"""Unit tests for Capability protocol and CapabilityContext."""

import pytest

from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.domain.results import TaskResult


def test_capability_context_creation():
    """Verify CapabilityContext holds execution_id and metadata."""
    ctx = CapabilityContext(execution_id="exec-123", metadata={"caller": "test"})
    assert ctx.execution_id == "exec-123"
    assert ctx.metadata["caller"] == "test"


def test_capability_context_defaults():
    """Verify default empty metadata for CapabilityContext."""
    ctx = CapabilityContext(execution_id="exec-456")
    assert ctx.execution_id == "exec-456"
    assert ctx.metadata == {}


def test_capability_context_is_frozen():
    """Verify CapabilityContext is immutable."""
    ctx = CapabilityContext(execution_id="exec-1")
    with pytest.raises(AttributeError):
        ctx.execution_id = "exec-2"  # type: ignore[misc]


def test_capability_protocol_satisfaction():
    """Verify a class implementing the protocol satisfies Capability."""

    class MockCapability:
        @property
        def capability_id(self) -> str:
            return "mock.capability"

        def execute(self, parameters, inputs, context) -> TaskResult:
            return TaskResult(output="mock_output")

    mock = MockCapability()
    assert isinstance(mock, Capability)
    assert mock.capability_id == "mock.capability"
    result = mock.execute({}, {}, CapabilityContext(execution_id="test"))
    assert result.output == "mock_output"


def test_capability_module_does_not_import_task():
    """Verify capabilities/base.py does not import Task."""
    import orchestration.capabilities.base as base_mod
    assert not hasattr(base_mod, "Task")
