"""Unit tests for Dependency value object."""

from orchestration.domain.dependencies import Dependency


def test_dependency_creation():
    """Verify Dependency is created with correct fields."""
    dep = Dependency(upstream_task_id="task-a", downstream_task_id="task-b")
    assert dep.upstream_task_id == "task-a"
    assert dep.downstream_task_id == "task-b"


def test_dependency_is_frozen():
    """Verify Dependency is immutable."""
    dep = Dependency(upstream_task_id="task-a", downstream_task_id="task-b")
    import pytest
    with pytest.raises(AttributeError):
        dep.upstream_task_id = "task-x"  # type: ignore[misc]


def test_dependency_equality():
    """Verify two Dependencies with the same fields are equal."""
    dep1 = Dependency(upstream_task_id="a", downstream_task_id="b")
    dep2 = Dependency(upstream_task_id="a", downstream_task_id="b")
    assert dep1 == dep2


def test_dependency_inequality():
    """Verify Dependencies with different fields are not equal."""
    dep1 = Dependency(upstream_task_id="a", downstream_task_id="b")
    dep2 = Dependency(upstream_task_id="a", downstream_task_id="c")
    assert dep1 != dep2
