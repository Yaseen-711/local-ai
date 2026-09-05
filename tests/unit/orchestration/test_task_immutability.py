"""Unit tests for Task declarative specification immutability."""

import pytest

from orchestration.domain.dependencies import Dependency
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import TaskStatus


def test_task_specification_fields_are_immutable_after_creation():
    task = Task(
        task_id="t-imm-1",
        plan_id="p-imm-1",
        title="Immutable Task",
        capability_id="test.echo",
        description="Original description",
        parameters={"param1": "val1"},
        input_references={"in1": DataReference(source_task_id="t-0", key="data")},
        dependencies=[Dependency(upstream_task_id="t-0", downstream_task_id="t-imm-1")],
    )

    # All specification fields should raise ValueError on reassignment
    with pytest.raises(ValueError, match="Cannot mutate declarative specification field 'task_id'"):
        task.task_id = "t-changed"

    with pytest.raises(ValueError, match="Cannot mutate declarative specification field 'capability_id'"):
        task.capability_id = "other.cap"

    with pytest.raises(ValueError, match="Cannot mutate declarative specification field 'plan_id'"):
        task.plan_id = "p-other"

    with pytest.raises(ValueError, match="Cannot mutate declarative specification field 'title'"):
        task.title = "Changed Title"

    with pytest.raises(ValueError, match="Cannot mutate declarative specification field 'description'"):
        task.description = "Changed Description"

    with pytest.raises(ValueError, match="Cannot mutate declarative specification field 'parameters'"):
        task.parameters = {"param2": "val2"}

    with pytest.raises(ValueError, match="Cannot mutate declarative specification field 'input_references'"):
        task.input_references = {}

    with pytest.raises(ValueError, match="Cannot mutate declarative specification field 'dependencies'"):
        task.dependencies = []


def test_task_execution_fields_remain_mutable_via_lifecycle():
    task = Task(
        task_id="t-exec-1",
        plan_id="p-exec-1",
        title="Execution Task",
        capability_id="test.echo",
        status=TaskStatus.READY,
    )

    # Lifecycle state transitions must work normally
    att = task.start_attempt("att-1")
    assert task.status == TaskStatus.RUNNING
    assert task.started_at is not None
    assert len(task.attempts) == 1

    res = TaskResult(output={"done": True})
    task.complete_attempt("att-1", res)
    assert task.status == TaskStatus.COMPLETED
    assert task.completed_at is not None
    assert task.result == res


def test_task_nested_collections_are_immutable():
    """Verify that in-place mutation of parameters, input_references, and dependencies is blocked."""
    task = Task(
        task_id="t-nested-1",
        plan_id="p-nested-1",
        title="Nested Immutability Task",
        capability_id="test.echo",
        parameters={"key1": "val1"},
        input_references={"in1": DataReference(source_task_id="t-0", key="data")},
        dependencies=[Dependency(upstream_task_id="t-0", downstream_task_id="t-nested-1")],
    )

    # 1. parameters in-place mutations must raise ValueError
    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'parameters'"):
        task.parameters["key2"] = "val2"

    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'parameters'"):
        task.parameters.pop("key1")

    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'parameters'"):
        task.parameters.update({"key3": "val3"})

    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'parameters'"):
        task.parameters.clear()

    # 2. input_references in-place mutations must raise ValueError
    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'input_references'"):
        task.input_references["in2"] = DataReference(source_task_id="t-1", key="data2")

    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'input_references'"):
        task.input_references.clear()

    # 3. dependencies in-place mutations must raise ValueError
    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'dependencies'"):
        task.dependencies.append(Dependency(upstream_task_id="t-2", downstream_task_id="t-nested-1"))

    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'dependencies'"):
        task.dependencies.extend([Dependency(upstream_task_id="t-3", downstream_task_id="t-nested-1")])

    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'dependencies'"):
        task.dependencies.pop()

    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'dependencies'"):
        task.dependencies.clear()

    # Read access remains normal dict/list operations
    assert task.parameters["key1"] == "val1"
    assert "key1" in task.parameters
    assert len(task.parameters) == 1
    assert len(task.dependencies) == 1
    assert task.dependencies[0].upstream_task_id == "t-0"


def test_task_immutability_persistence_roundtrip():
    """Verify that immutable tasks can be serialized and deserialized via mappers without issues."""
    from orchestration.persistence.mappers import task_to_model, model_to_task

    original_task = Task(
        task_id="t-persist-1",
        plan_id="p-persist-1",
        title="Persist Task",
        capability_id="test.echo",
        parameters={"alpha": 1, "beta": "two"},
        input_references={"in1": DataReference(source_task_id="t-0", key="ref")},
        dependencies=[Dependency(upstream_task_id="t-0", downstream_task_id="t-persist-1")],
    )

    model = task_to_model(original_task)
    restored = model_to_task(model, dependencies=list(original_task.dependencies))

    assert restored.task_id == original_task.task_id
    assert restored.parameters["alpha"] == 1
    assert restored.dependencies[0].upstream_task_id == "t-0"

    # Restored task collections must also be immutable proxies
    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'parameters'"):
        restored.parameters["new"] = True

    with pytest.raises(ValueError, match="Cannot mutate Task specification collection 'dependencies'"):
        restored.dependencies.append(Dependency(upstream_task_id="t-x", downstream_task_id="t-persist-1"))
