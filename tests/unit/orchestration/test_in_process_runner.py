"""Unit tests for InProcessPlanRunner."""

import pytest

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.dependencies import Dependency
from orchestration.domain.plans import Plan
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import (
    AttemptStatus,
    PlanStatus,
    TaskErrorCategory,
    TaskStatus,
)
from orchestration.execution.runner import InProcessPlanRunner


class UppercaseCapability:
    """Test capability that converts 'text' input to uppercase."""

    @property
    def capability_id(self) -> str:
        return "test.uppercase"

    def execute(self, parameters, inputs, context: CapabilityContext) -> TaskResult:
        text = inputs.get("text") or parameters.get("text", "")
        return TaskResult(output=text.upper(), metadata={"exec_id": context.execution_id})


class DictOutputCapability:
    """Test capability returning a structured dict output."""

    @property
    def capability_id(self) -> str:
        return "test.dict_producer"

    def execute(self, parameters, inputs, context: CapabilityContext) -> TaskResult:
        return TaskResult(
            output={
                "summary": "Financial growth observed",
                "score": 95,
            }
        )


class FailingCapability:
    """Test capability that unconditionally raises a RuntimeError."""

    @property
    def capability_id(self) -> str:
        return "test.failing"

    def execute(self, parameters, inputs, context: CapabilityContext) -> TaskResult:
        raise RuntimeError("External service failure")


# ---------------------------------------------------------------------------
# Runner Execution Tests
# ---------------------------------------------------------------------------

def test_runner_executes_single_task():
    """Verify runner activates draft plan, executes task, records attempt, and completes."""
    registry = CapabilityRegistry()
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-1", goal_id="g-1", title="Single Task Plan")
    task = Task(
        task_id="t-1",
        plan_id="p-1",
        title="Uppercase task",
        capability_id="test.uppercase",
        parameters={"text": "hello world"},
    )
    plan.add_task(task)

    runner = InProcessPlanRunner(registry)
    completed_plan = runner.run(plan)

    assert completed_plan.status == PlanStatus.COMPLETED
    assert task.status == TaskStatus.COMPLETED
    assert task.result is not None
    assert task.result.output == "HELLO WORLD"
    assert len(task.attempts) == 1
    assert task.attempts[0].status == AttemptStatus.SUCCESS
    assert task.attempts[0].result is task.result


def test_runner_executes_linear_chain_with_data_flow():
    """Verify Task A output flows into Task B via input_references."""
    registry = CapabilityRegistry()
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-chain", goal_id="g-1", title="Chain")
    task_a = Task(
        task_id="a",
        plan_id="p-chain",
        title="First step",
        capability_id="test.uppercase",
        parameters={"text": "first step text"},
    )
    task_b = Task(
        task_id="b",
        plan_id="p-chain",
        title="Second step",
        capability_id="test.uppercase",
        dependencies=[Dependency(upstream_task_id="a", downstream_task_id="b")],
        input_references={
            "text": DataReference(key="output", source_task_id="a"),
        },
    )
    plan.add_task(task_a)
    plan.add_task(task_b)

    runner = InProcessPlanRunner(registry)
    completed_plan = runner.run(plan)

    assert completed_plan.status == PlanStatus.COMPLETED
    assert task_a.status == TaskStatus.COMPLETED
    assert task_b.status == TaskStatus.COMPLETED
    assert task_a.result.output == "FIRST STEP TEXT"
    assert task_b.result.output == "FIRST STEP TEXT"


def test_runner_resolves_dict_key_reference():
    """Verify input_reference resolving a specific key from upstream dict output."""
    registry = CapabilityRegistry()
    registry.register(DictOutputCapability())
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-dict", goal_id="g-1", title="Dict Ref Plan")
    task_producer = Task(
        task_id="t-prod",
        plan_id="p-dict",
        title="Producer",
        capability_id="test.dict_producer",
    )
    task_consumer = Task(
        task_id="t-cons",
        plan_id="p-dict",
        title="Consumer",
        capability_id="test.uppercase",
        dependencies=[Dependency(upstream_task_id="t-prod", downstream_task_id="t-cons")],
        input_references={
            "text": DataReference(key="summary", source_task_id="t-prod"),
        },
    )
    plan.add_task(task_producer)
    plan.add_task(task_consumer)

    runner = InProcessPlanRunner(registry)
    runner.run(plan)

    assert plan.status == PlanStatus.COMPLETED
    assert task_consumer.status == TaskStatus.COMPLETED
    # UppercaseCapability should receive the "summary" string from dict
    assert task_consumer.result.output == "FINANCIAL GROWTH OBSERVED"


def test_runner_diamond_dag_execution():
    """Verify diamond DAG: A -> B, A -> C, (B, C) -> D."""
    registry = CapabilityRegistry()
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-diamond", goal_id="g-1", title="Diamond Plan")
    ta = Task(task_id="a", plan_id="p-diamond", title="A", capability_id="test.uppercase", parameters={"text": "a"})
    tb = Task(task_id="b", plan_id="p-diamond", title="B", capability_id="test.uppercase",
              dependencies=[Dependency("a", "b")], parameters={"text": "b"})
    tc = Task(task_id="c", plan_id="p-diamond", title="C", capability_id="test.uppercase",
              dependencies=[Dependency("a", "c")], parameters={"text": "c"})
    td = Task(task_id="d", plan_id="p-diamond", title="D", capability_id="test.uppercase",
              dependencies=[Dependency("b", "d"), Dependency("c", "d")], parameters={"text": "d"})

    for t in (ta, tb, tc, td):
        plan.add_task(t)

    runner = InProcessPlanRunner(registry)
    runner.run(plan)

    assert plan.status == PlanStatus.COMPLETED
    for t in (ta, tb, tc, td):
        assert t.status == TaskStatus.COMPLETED


def test_runner_missing_capability_fails_task_and_plan():
    """Verify unresolvable capability marks attempt/task FAILED and blocks downstream."""
    registry = CapabilityRegistry()  # Empty registry

    plan = Plan(plan_id="p-missing", goal_id="g-1", title="Missing Cap Plan")
    t1 = Task(task_id="t1", plan_id="p-missing", title="T1", capability_id="unregistered.cap")
    t2 = Task(task_id="t2", plan_id="p-missing", title="T2", capability_id="unregistered.cap",
              dependencies=[Dependency("t1", "t2")])
    plan.add_task(t1)
    plan.add_task(t2)

    runner = InProcessPlanRunner(registry)
    runner.run(plan)

    assert plan.status == PlanStatus.FAILED
    assert t1.status == TaskStatus.FAILED
    assert t1.error is not None
    assert t1.error.category == TaskErrorCategory.CAPABILITY
    assert t1.error.error_code == "CAPABILITY_NOT_FOUND"
    assert t2.status == TaskStatus.BLOCKED


def test_runner_capability_exception_fails_task_and_plan():
    """Verify exception inside capability maps to TaskError and marks task FAILED."""
    registry = CapabilityRegistry()
    registry.register(FailingCapability())

    plan = Plan(plan_id="p-fail", goal_id="g-1", title="Failing Plan")
    t1 = Task(task_id="t1", plan_id="p-fail", title="T1", capability_id="test.failing")
    t2 = Task(task_id="t2", plan_id="p-fail", title="T2", capability_id="test.failing",
              dependencies=[Dependency("t1", "t2")])
    plan.add_task(t1)
    plan.add_task(t2)

    runner = InProcessPlanRunner(registry)
    runner.run(plan)

    assert plan.status == PlanStatus.FAILED
    assert t1.status == TaskStatus.FAILED
    assert t1.error is not None
    assert t1.error.category == TaskErrorCategory.EXECUTION
    assert "External service failure" in t1.error.message
    assert t2.status == TaskStatus.BLOCKED


def test_runner_empty_plan_raises():
    """Verify running a plan with no tasks raises ValueError."""
    registry = CapabilityRegistry()
    plan = Plan(plan_id="p-empty", goal_id="g-1", title="Empty Plan")
    runner = InProcessPlanRunner(registry)

    with pytest.raises(ValueError, match="has no tasks"):
        runner.run(plan)


def test_runner_already_completed_plan_raises():
    """Verify running an already completed plan raises ValueError."""
    registry = CapabilityRegistry()
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-done", goal_id="g-1", title="Done Plan")
    plan.add_task(Task(task_id="t1", plan_id="p-done", title="T1", capability_id="test.uppercase", parameters={"text": "x"}))
    runner = InProcessPlanRunner(registry)
    runner.run(plan)
    assert plan.status == PlanStatus.COMPLETED

    # Running again should raise
    with pytest.raises(ValueError, match="expected 'active'"):
        runner.run(plan)


def test_runner_unresolvable_source_task_fails_task_honestly():
    """Verify missing source_task_id raises ValueError, mapped to TaskError, failing task."""
    registry = CapabilityRegistry()
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-bad-source", goal_id="g-1", title="Bad Source Plan")
    task = Task(
        task_id="t1",
        plan_id="p-bad-source",
        title="Bad Source Task",
        capability_id="test.uppercase",
        input_references={
            "text": DataReference(key="out", source_task_id="nonexistent_task"),
        },
    )
    plan.add_task(task)

    runner = InProcessPlanRunner(registry)
    runner.run(plan)

    assert plan.status == PlanStatus.FAILED
    assert task.status == TaskStatus.FAILED
    assert task.error is not None
    assert task.error.category == TaskErrorCategory.EXECUTION
    assert "nonexistent_task" in task.error.message


def test_runner_reference_without_source_or_uri_fails_honestly():
    """Verify input reference with neither source_task_id nor uri fails task honestly."""
    registry = CapabilityRegistry()
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-empty-ref", goal_id="g-1", title="Empty Ref Plan")
    task = Task(
        task_id="t1",
        plan_id="p-empty-ref",
        title="Empty Ref Task",
        capability_id="test.uppercase",
        input_references={
            "text": DataReference(key="out"),  # neither source_task_id nor uri
        },
    )
    plan.add_task(task)

    runner = InProcessPlanRunner(registry)
    runner.run(plan)

    assert plan.status == PlanStatus.FAILED
    assert task.status == TaskStatus.FAILED
    assert task.error is not None
    assert task.error.category == TaskErrorCategory.EXECUTION
    assert "neither 'source_task_id' nor 'uri'" in task.error.message


def test_in_process_runner_lifecycle_start_and_wait():
    """Verify decoupled start and wait execution."""
    registry = CapabilityRegistry()
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-lifecycle", goal_id="g-1", title="Lifecycle Plan")
    plan.add_task(
        Task(
            task_id="t1",
            plan_id="p-lifecycle",
            title="T1",
            capability_id="test.uppercase",
            parameters={"text": "hello lifecycle"},
        )
    )

    runner = InProcessPlanRunner(registry)
    handle = runner.start(plan)
    assert handle.plan_id == "p-lifecycle"
    assert handle.execution_id.startswith("exec-p-lifecycle")

    # Status before wait
    status_snapshot = runner.get_status(handle)
    assert status_snapshot.plan_id == "p-lifecycle"
    assert status_snapshot.status == PlanStatus.ACTIVE
    assert not status_snapshot.is_terminal

    # Wait
    result = runner.wait(handle)
    assert result.status == PlanStatus.COMPLETED
    assert "t1" in result.task_results
    assert result.task_results["t1"].output == "HELLO LIFECYCLE"

    # Status after wait
    status_after = runner.get_status(handle)
    assert status_after.status == PlanStatus.COMPLETED
    assert status_after.is_terminal


def test_in_process_runner_lifecycle_cancel():
    """Verify cancelling an in-flight execution."""
    registry = CapabilityRegistry()
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-cancel-run", goal_id="g-1", title="Cancel Plan")
    plan.add_task(
        Task(
            task_id="t1",
            plan_id="p-cancel-run",
            title="T1",
            capability_id="test.uppercase",
            parameters={"text": "cancel me"},
        )
    )

    runner = InProcessPlanRunner(registry)
    handle = runner.start(plan)
    runner.cancel(handle)

    status_snapshot = runner.get_status(handle)
    assert status_snapshot.status == PlanStatus.CANCELLED
    assert status_snapshot.is_terminal
    assert plan.status == PlanStatus.CANCELLED
    assert plan.tasks["t1"].status == TaskStatus.CANCELLED


def test_runner_fails_when_ref_key_missing_from_dict():
    """Verify input_reference resolution raises ValueError and fails task when ref.key is missing from dict output."""
    registry = CapabilityRegistry()
    registry.register(DictOutputCapability())
    registry.register(UppercaseCapability())

    plan = Plan(plan_id="p-missing-key", goal_id="g-1", title="Missing Key Plan")
    task_producer = Task(
        task_id="t-prod",
        plan_id="p-missing-key",
        title="Producer",
        capability_id="test.dict_producer",
    )
    task_consumer = Task(
        task_id="t-cons",
        plan_id="p-missing-key",
        title="Consumer",
        capability_id="test.uppercase",
        dependencies=[Dependency(upstream_task_id="t-prod", downstream_task_id="t-cons")],
        input_references={
            "text": DataReference(key="nonexistent_field", source_task_id="t-prod"),
        },
    )
    plan.add_task(task_producer)
    plan.add_task(task_consumer)

    runner = InProcessPlanRunner(registry)
    completed_plan = runner.run(plan)

    assert completed_plan.status == PlanStatus.FAILED
    assert task_producer.status == TaskStatus.COMPLETED
    assert task_consumer.status == TaskStatus.FAILED
    assert task_consumer.error is not None
    assert "nonexistent_field" in task_consumer.error.message
    assert "not found in output" in task_consumer.error.message
