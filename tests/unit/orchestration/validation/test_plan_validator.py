"""Unit tests for deterministic 4-stage PlanValidator."""

import pytest

from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.dependencies import Dependency
from orchestration.domain.plans import Plan
from orchestration.domain.references import DataReference
from orchestration.domain.tasks import Task
from orchestration.validation import (
    PlanValidator,
    ValidationError,
    ValidationResult,
    ValidationStage,
)


class DummyCapability:
    def __init__(self, cid: str):
        self._cid = cid

    @property
    def capability_id(self) -> str:
        return self._cid

    def execute(self, parameters, inputs, context):
        return None


@pytest.fixture
def registry_with_capabilities():
    registry = CapabilityRegistry()

    # Simple capability
    registry.register(
        DummyCapability("summarize"),
        descriptor=CapabilityDescriptor(
            capability_id="summarize",
            description="Summarizes text",
            parameter_schema={"type": "object", "required": ["text"]},
        ),
    )

    # Transform capability
    registry.register(
        DummyCapability("transform"),
        descriptor=CapabilityDescriptor(
            capability_id="transform",
            description="Transforms text",
        ),
    )

    # Deprecated capability
    registry.register(
        DummyCapability("old_cap"),
        descriptor=CapabilityDescriptor(
            capability_id="old_cap",
            description="Old capability",
            is_deprecated=True,
            deprecation_reason="Use transform instead",
        ),
    )

    return registry


def test_valid_plan_with_parallel_disconnected_components(registry_with_capabilities):
    """LOCKED DECISION: Disconnected components (parallel independent tasks) must be accepted."""
    validator = PlanValidator(capability_registry=registry_with_capabilities)

    # Plan with:
    # Component 1: t1 -> t2
    # Component 2: t3 (completely independent, parallel)
    plan = Plan(plan_id="p1", goal_id="g1", title="Parallel Plan")

    t1 = Task(
        task_id="t1",
        plan_id="p1",
        capability_id="transform",
        title="Transform Step 1",
    )
    t2 = Task(
        task_id="t2",
        plan_id="p1",
        capability_id="summarize",
        title="Summarize Step 2",
        parameters={"text": "content"},
        dependencies=[Dependency("t1", "t2")],
    )
    t3 = Task(
        task_id="t3",
        plan_id="p1",
        capability_id="transform",
        title="Independent Task 3",
    )

    plan.add_task(t1)
    plan.add_task(t2)
    plan.add_task(t3)

    result = validator.validate(plan)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_structural_empty_plan_fails(registry_with_capabilities):
    validator = PlanValidator(capability_registry=registry_with_capabilities)
    plan = Plan(plan_id="p1", goal_id="g1", title="Empty Plan")

    result = validator.validate(plan)
    assert result.is_valid is False
    assert any(e.code == "EMPTY_PLAN" for e in result.errors)


def test_structural_cycle_detection(registry_with_capabilities):
    validator = PlanValidator(capability_registry=registry_with_capabilities)
    plan = Plan(plan_id="p1", goal_id="g1", title="Cyclic Plan")

    t1 = Task(
        task_id="t1",
        plan_id="p1",
        capability_id="transform",
        title="T1",
        dependencies=[Dependency("t2", "t1")],
    )
    t2 = Task(
        task_id="t2",
        plan_id="p1",
        capability_id="transform",
        title="T2",
        dependencies=[Dependency("t1", "t2")],
    )

    plan.add_task(t1)
    plan.add_task(t2)

    result = validator.validate(plan)
    assert result.is_valid is False
    assert any(e.code == "DEPENDENCY_CYCLE" for e in result.errors)


def test_structural_self_dependency_fails(registry_with_capabilities):
    validator = PlanValidator(capability_registry=registry_with_capabilities)
    plan = Plan(plan_id="p1", goal_id="g1", title="Self Dep Plan")

    t1 = Task(
        task_id="t1",
        plan_id="p1",
        capability_id="transform",
        title="T1",
        dependencies=[Dependency("t1", "t1")],
    )
    plan.add_task(t1)

    result = validator.validate(plan)
    assert result.is_valid is False
    assert any(e.code == "SELF_DEPENDENCY" for e in result.errors)


def test_structural_unknown_dependency_fails(registry_with_capabilities):
    validator = PlanValidator(capability_registry=registry_with_capabilities)
    plan = Plan(plan_id="p1", goal_id="g1", title="Missing Dep Plan")

    t1 = Task(
        task_id="t1",
        plan_id="p1",
        capability_id="transform",
        title="T1",
        dependencies=[Dependency("t_nonexistent", "t1")],
    )
    plan.add_task(t1)

    result = validator.validate(plan)
    assert result.is_valid is False
    assert any(e.code == "UNKNOWN_UPSTREAM_TASK" for e in result.errors)


def test_capability_validation_unknown_and_missing_params(registry_with_capabilities):
    validator = PlanValidator(capability_registry=registry_with_capabilities)
    plan = Plan(plan_id="p1", goal_id="g1", title="Capability Test Plan")

    # t1 uses unknown capability
    t1 = Task(
        task_id="t1",
        plan_id="p1",
        capability_id="quantum_compute",
        title="T1",
    )
    # t2 uses summarize without required parameter 'text'
    t2 = Task(
        task_id="t2",
        plan_id="p1",
        capability_id="summarize",
        title="T2",
        parameters={},  # Missing 'text'
    )

    plan.add_task(t1)
    plan.add_task(t2)

    result = validator.validate(plan)
    assert result.is_valid is False
    assert any(e.code == "UNKNOWN_CAPABILITY" for e in result.errors)
    assert any(e.code == "MISSING_REQUIRED_PARAMETER" for e in result.errors)


def test_capability_warning_on_deprecated(registry_with_capabilities):
    validator = PlanValidator(capability_registry=registry_with_capabilities)
    plan = Plan(plan_id="p1", goal_id="g1", title="Deprecated Plan")

    t1 = Task(
        task_id="t1",
        plan_id="p1",
        capability_id="old_cap",
        title="T1",
    )
    plan.add_task(t1)

    result = validator.validate(plan)
    assert result.is_valid is True  # Deprecation produces warning, not failure
    assert len(result.warnings) > 0
    assert "deprecated" in result.warnings[0].lower()


def test_hard_constraints_limits(registry_with_capabilities):
    validator = PlanValidator(
        capability_registry=registry_with_capabilities,
        max_tasks=2,
        max_depth=2,
    )

    # Exceed max_tasks
    p_tasks = Plan(plan_id="p_tasks", goal_id="g1", title="Too Many Tasks")
    p_tasks.add_task(Task(task_id="t1", plan_id="p_tasks", title="T1", capability_id="transform"))
    p_tasks.add_task(Task(task_id="t2", plan_id="p_tasks", title="T2", capability_id="transform"))
    p_tasks.add_task(Task(task_id="t3", plan_id="p_tasks", title="T3", capability_id="transform"))

    res_tasks = validator.validate(p_tasks)
    assert res_tasks.is_valid is False
    assert any(e.code == "EXCEEDED_MAX_TASKS" for e in res_tasks.errors)

    # Exceed max_depth: t1 -> t2 -> t3 (depth 3 > max_depth 2)
    p_depth = Plan(plan_id="p_depth", goal_id="g1", title="Too Deep Plan")
    p_depth.add_task(Task(task_id="t1", plan_id="p_depth", title="T1", capability_id="transform"))
    p_depth.add_task(Task(task_id="t2", plan_id="p_depth", title="T2", capability_id="transform", dependencies=[Dependency("t1", "t2")]))
    p_depth.add_task(Task(task_id="t3", plan_id="p_depth", title="T3", capability_id="transform", dependencies=[Dependency("t2", "t3")]))

    # Use validator with max_tasks=10, max_depth=2
    val_depth = PlanValidator(capability_registry=registry_with_capabilities, max_tasks=10, max_depth=2)
    res_depth = val_depth.validate(p_depth)
    assert res_depth.is_valid is False
    assert any(e.code == "EXCEEDED_MAX_DEPTH" for e in res_depth.errors)


def test_feasibility_data_references_available_vs_execution_produced(registry_with_capabilities):
    """LOCKED DECISION:
    Distinguish already-available data from execution-produced data.
    Ordering required only for execution-produced data.
    """
    validator = PlanValidator(capability_registry=registry_with_capabilities)

    # Case A: Execution-produced data WITHOUT ordering dependency -> FAILS
    p_unordered = Plan(plan_id="p_a", goal_id="g1", title="Unordered Data Flow")
    t1 = Task(task_id="t1", plan_id="p_a", title="Producer", capability_id="transform")
    t2 = Task(
        task_id="t2",
        plan_id="p_a",
        title="Consumer",
        capability_id="transform",
        input_references={"input_data": DataReference(key="input_data", source_task_id="t1")},
        # Missing Dependency("t1", "t2")
    )
    p_unordered.add_task(t1)
    p_unordered.add_task(t2)

    res_unordered = validator.validate(p_unordered)
    assert res_unordered.is_valid is False
    assert any(e.code == "UNORDERED_DATA_REFERENCE" for e in res_unordered.errors)

    # Case B: Execution-produced data WITH ordering dependency -> PASSES
    p_ordered = Plan(plan_id="p_b", goal_id="g1", title="Ordered Data Flow")
    t1_b = Task(task_id="t1", plan_id="p_b", title="Producer", capability_id="transform")
    t2_b = Task(
        task_id="t2",
        plan_id="p_b",
        title="Consumer",
        capability_id="transform",
        input_references={"input_data": DataReference(key="input_data", source_task_id="t1")},
        dependencies=[Dependency("t1", "t2")],
    )
    p_ordered.add_task(t1_b)
    p_ordered.add_task(t2_b)

    res_ordered = validator.validate(p_ordered)
    assert res_ordered.is_valid is True

    # Case C: Already-available data (from prior completed task) WITHOUT ordering dependency -> PASSES
    p_avail = Plan(plan_id="p_c", goal_id="g1", title="Available Data Reference")
    t_consumer = Task(
        task_id="t_consumer",
        plan_id="p_c",
        title="Consumer",
        capability_id="transform",
        input_references={"prior_data": DataReference(key="prior_data", source_task_id="t_prior_completed")},
    )
    p_avail.add_task(t_consumer)

    # Validate with available_task_ids containing 't_prior_completed'
    res_avail = validator.validate(p_avail, available_task_ids={"t_prior_completed"})
    assert res_avail.is_valid is True

    # Case D: Unknown data source (neither in plan nor available) -> FAILS
    res_unknown = validator.validate(p_avail, available_task_ids=set())
    assert res_unknown.is_valid is False
    assert any(e.code == "UNKNOWN_DATA_SOURCE" for e in res_unknown.errors)
