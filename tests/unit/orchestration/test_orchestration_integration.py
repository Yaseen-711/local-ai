"""Integration-style tests for full orchestration workflows.

Tests multi-entity interactions: Goal → Plan → Task → Attempt → Result/Error,
including DAG dependency propagation and readiness evaluation across
multiple tasks.
"""

import pytest

from orchestration.domain.attempts import Attempt
from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan, PlanRevision
from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import (
    AttemptStatus,
    GoalStatus,
    PlanStatus,
    TaskErrorCategory,
    TaskStatus,
)


def test_full_successful_workflow():
    """End-to-end: Goal → Plan → Tasks → Attempts → Success → Completed."""
    # 1. Create goal
    goal = Goal(goal_id="g-1", description="Analyze document")
    assert goal.status == GoalStatus.PENDING

    # 2. Create plan with two tasks: extract → analyze
    plan = Plan(plan_id="p-1", goal_id="g-1", title="Document Analysis Plan")

    task_extract = Task(
        task_id="t-extract",
        plan_id="p-1",
        title="Extract text",
        capability_id="workflow.text_extraction",
    )
    task_analyze = Task(
        task_id="t-analyze",
        plan_id="p-1",
        title="Analyze text",
        capability_id="inference.chat",
        dependencies=[
            Dependency(upstream_task_id="t-extract", downstream_task_id="t-analyze"),
        ],
        input_references={
            "source": DataReference(key="extracted_text", source_task_id="t-extract"),
        },
    )
    plan.add_task(task_extract)
    plan.add_task(task_analyze)

    # 3. Record initial revision and activate
    plan.record_revision("rev-1", "Initial plan")
    plan.activate()
    goal.activate(plan_id="p-1")

    assert plan.status == PlanStatus.ACTIVE
    assert goal.status == GoalStatus.ACTIVE

    # 4. Resolve readiness for task_extract (no deps → READY)
    task_extract.update_readiness({})
    assert task_extract.status == TaskStatus.READY

    # task_analyze stays PENDING (upstream not completed yet)
    task_analyze.update_readiness({"t-extract": task_extract.status})
    assert task_analyze.status == TaskStatus.PENDING

    # 5. Execute task_extract
    task_extract.start_attempt("a-1")
    task_extract.complete_attempt("a-1", TaskResult(output="Extracted text content"))
    assert task_extract.status == TaskStatus.COMPLETED

    # 6. Resolve readiness for task_analyze (upstream completed → READY)
    task_analyze.update_readiness({"t-extract": task_extract.status})
    assert task_analyze.status == TaskStatus.READY

    # 7. Execute task_analyze
    task_analyze.start_attempt("a-2")
    task_analyze.complete_attempt("a-2", TaskResult(
        output={"summary": "The document discusses..."},
        artifacts=[ArtifactReference(
            artifact_id="art-1",
            name="analysis_report.json",
            uri="mem://results/art-1",
            mime_type="application/json",
        )],
    ))
    assert task_analyze.status == TaskStatus.COMPLETED

    # 8. Mark plan and goal completed
    plan.mark_completed()
    goal.mark_completed()
    assert plan.status == PlanStatus.COMPLETED
    assert goal.status == GoalStatus.COMPLETED


def test_failure_propagation_through_dag():
    """Verify upstream failure blocks downstream tasks."""
    plan = Plan(plan_id="p-1", goal_id="g-1", title="Pipeline")
    task_a = Task(
        task_id="a", plan_id="p-1", title="Step A",
        capability_id="test.a",
    )
    task_b = Task(
        task_id="b", plan_id="p-1", title="Step B",
        capability_id="test.b",
        dependencies=[Dependency(upstream_task_id="a", downstream_task_id="b")],
    )
    task_c = Task(
        task_id="c", plan_id="p-1", title="Step C",
        capability_id="test.c",
        dependencies=[Dependency(upstream_task_id="b", downstream_task_id="c")],
    )
    plan.add_task(task_a)
    plan.add_task(task_b)
    plan.add_task(task_c)
    plan.activate()

    # A becomes READY, starts, fails
    task_a.update_readiness({})
    task_a.start_attempt("a-1")
    task_a.fail_attempt("a-1", TaskError(
        message="Model OOM", category=TaskErrorCategory.INFRASTRUCTURE,
    ))
    assert task_a.status == TaskStatus.FAILED

    # B should become BLOCKED
    task_b.update_readiness({"a": task_a.status})
    assert task_b.status == TaskStatus.BLOCKED

    # C should also become BLOCKED (transitively)
    task_c.update_readiness({"b": task_b.status})
    assert task_c.status == TaskStatus.BLOCKED

    # Plan fails
    plan.mark_failed()
    assert plan.status == PlanStatus.FAILED


def test_diamond_dag_readiness():
    """Verify diamond DAG readiness: D waits for both B and C."""
    # A → B, A → C, B → D, C → D
    plan = Plan(plan_id="p-1", goal_id="g-1", title="Diamond")
    tasks = {
        "a": Task(task_id="a", plan_id="p-1", title="A", capability_id="t.a"),
        "b": Task(task_id="b", plan_id="p-1", title="B", capability_id="t.b",
                  dependencies=[Dependency("a", "b")]),
        "c": Task(task_id="c", plan_id="p-1", title="C", capability_id="t.c",
                  dependencies=[Dependency("a", "c")]),
        "d": Task(task_id="d", plan_id="p-1", title="D", capability_id="t.d",
                  dependencies=[Dependency("b", "d"), Dependency("c", "d")]),
    }
    for t in tasks.values():
        plan.add_task(t)
    plan.activate()

    statuses = lambda: {tid: t.status for tid, t in tasks.items()}

    # A → READY → RUNNING → COMPLETED
    tasks["a"].update_readiness({})
    tasks["a"].start_attempt("aa")
    tasks["a"].complete_attempt("aa", TaskResult())

    # B and C both become READY
    tasks["b"].update_readiness(statuses())
    tasks["c"].update_readiness(statuses())
    assert tasks["b"].status == TaskStatus.READY
    assert tasks["c"].status == TaskStatus.READY

    # D is still PENDING (B and C not completed)
    tasks["d"].update_readiness(statuses())
    assert tasks["d"].status == TaskStatus.PENDING

    # Complete B but not C
    tasks["b"].start_attempt("ba")
    tasks["b"].complete_attempt("ba", TaskResult())
    tasks["d"].update_readiness(statuses())
    assert tasks["d"].status == TaskStatus.PENDING  # Still waiting for C

    # Complete C
    tasks["c"].start_attempt("ca")
    tasks["c"].complete_attempt("ca", TaskResult())
    tasks["d"].update_readiness(statuses())
    assert tasks["d"].status == TaskStatus.READY  # Both B and C done


def test_cancellation_cascade():
    """Verify cancelling a plan cancels the goal pathway."""
    goal = Goal(goal_id="g-1", description="Test")
    plan = Plan(plan_id="p-1", goal_id="g-1", title="Test Plan")
    task = Task(task_id="t-1", plan_id="p-1", title="T", capability_id="test")
    plan.add_task(task)
    plan.activate()
    goal.activate("p-1")

    plan.cancel()
    goal.cancel()
    assert plan.status == PlanStatus.CANCELLED
    assert goal.status == GoalStatus.CANCELLED


def test_orchestration_imports_from_top_level():
    """Verify all domain entities are importable from orchestration package."""
    from orchestration import (
        Goal,
        Plan,
        PlanRevision,
        Task,
        Dependency,
        Attempt,
        TaskResult,
        TaskError,
        DataReference,
        ArtifactReference,
        GoalStatus,
        PlanStatus,
        TaskStatus,
        AttemptStatus,
        TaskErrorCategory,
    )
    # Just verify they're importable — no assertions needed beyond no ImportError
    assert Goal is not None
