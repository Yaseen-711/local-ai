"""Unit tests for Plan lifecycle, DAG validation, and PlanRevision."""

import pytest

from orchestration.domain.dependencies import Dependency
from orchestration.domain.plans import Plan, PlanRevision
from orchestration.domain.tasks import Task
from orchestration.domain.types import PlanStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id: str, plan_id: str, deps: list = None) -> Task:
    """Create a minimal Task for plan tests."""
    return Task(
        task_id=task_id,
        plan_id=plan_id,
        title=f"Task {task_id}",
        capability_id="test.capability",
        dependencies=deps or [],
    )


# ---------------------------------------------------------------------------
# Plan Creation & Task Management
# ---------------------------------------------------------------------------

def test_plan_creation_defaults():
    """Verify Plan is created with DRAFT status and sensible defaults."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    assert plan.plan_id == "plan-1"
    assert plan.goal_id == "goal-1"
    assert plan.status == PlanStatus.DRAFT
    assert plan.tasks == {}
    assert plan.revisions == []
    assert plan.completed_at is None


def test_plan_add_task():
    """Verify adding tasks to a DRAFT plan."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    task = _make_task("task-a", "plan-1")
    plan.add_task(task)
    assert "task-a" in plan.tasks
    assert plan.tasks["task-a"] is task


def test_plan_add_task_wrong_plan_id_raises():
    """Verify adding a task with mismatched plan_id raises ValueError."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    task = _make_task("task-a", "plan-wrong")
    with pytest.raises(ValueError, match="expected 'plan-1'"):
        plan.add_task(task)


def test_plan_add_duplicate_task_raises():
    """Verify adding a task with duplicate ID raises ValueError."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))
    with pytest.raises(ValueError, match="already exists"):
        plan.add_task(_make_task("task-a", "plan-1"))


def test_plan_add_task_to_completed_raises():
    """Verify adding tasks to a COMPLETED plan raises ValueError."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))
    plan.activate()
    plan.mark_completed()
    with pytest.raises(ValueError, match="plan status is 'completed'"):
        plan.add_task(_make_task("task-b", "plan-1"))


# ---------------------------------------------------------------------------
# Plan Activation
# ---------------------------------------------------------------------------

def test_plan_activate():
    """Verify DRAFT → ACTIVE transition with DAG validation."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))
    plan.activate()
    assert plan.status == PlanStatus.ACTIVE


def test_plan_activate_empty_raises():
    """Verify activating a plan with no tasks raises ValueError."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    with pytest.raises(ValueError, match="has no tasks"):
        plan.activate()


def test_plan_activate_not_draft_raises():
    """Verify activating a non-DRAFT plan raises ValueError."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))
    plan.activate()
    with pytest.raises(ValueError, match="expected 'draft'"):
        plan.activate()


# ---------------------------------------------------------------------------
# Plan Terminal Transitions
# ---------------------------------------------------------------------------

def test_plan_mark_completed():
    """Verify ACTIVE → COMPLETED transition."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))
    plan.activate()
    plan.mark_completed()
    assert plan.status == PlanStatus.COMPLETED
    assert plan.completed_at is not None


def test_plan_mark_completed_not_active_raises():
    """Verify completing a non-ACTIVE plan raises ValueError."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    with pytest.raises(ValueError, match="expected 'active'"):
        plan.mark_completed()


def test_plan_mark_failed():
    """Verify ACTIVE → FAILED transition."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))
    plan.activate()
    plan.mark_failed()
    assert plan.status == PlanStatus.FAILED
    assert plan.completed_at is not None


def test_plan_cancel():
    """Verify cancellation from DRAFT and ACTIVE states."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.cancel()
    assert plan.status == PlanStatus.CANCELLED

    plan2 = Plan(plan_id="plan-2", goal_id="goal-1", title="Test Plan 2")
    plan2.add_task(_make_task("task-a", "plan-2"))
    plan2.activate()
    plan2.cancel()
    assert plan2.status == PlanStatus.CANCELLED


def test_plan_cancel_completed_raises():
    """Verify cancelling a COMPLETED plan raises ValueError."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))
    plan.activate()
    plan.mark_completed()
    with pytest.raises(ValueError, match="already completed"):
        plan.cancel()


def test_plan_cancel_idempotent():
    """Verify cancelling an already-cancelled plan is a no-op."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.cancel()
    plan.cancel()  # Should not raise
    assert plan.status == PlanStatus.CANCELLED


# ---------------------------------------------------------------------------
# DAG Validation
# ---------------------------------------------------------------------------

def test_dag_valid_linear_chain():
    """Verify valid linear chain: A → B → C."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Linear")
    plan.add_task(_make_task("a", "plan-1"))
    plan.add_task(_make_task("b", "plan-1", deps=[
        Dependency(upstream_task_id="a", downstream_task_id="b"),
    ]))
    plan.add_task(_make_task("c", "plan-1", deps=[
        Dependency(upstream_task_id="b", downstream_task_id="c"),
    ]))
    plan.validate_dag()  # Should not raise


def test_dag_valid_diamond():
    """Verify valid diamond: A → B, A → C, B → D, C → D."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Diamond")
    plan.add_task(_make_task("a", "plan-1"))
    plan.add_task(_make_task("b", "plan-1", deps=[
        Dependency(upstream_task_id="a", downstream_task_id="b"),
    ]))
    plan.add_task(_make_task("c", "plan-1", deps=[
        Dependency(upstream_task_id="a", downstream_task_id="c"),
    ]))
    plan.add_task(_make_task("d", "plan-1", deps=[
        Dependency(upstream_task_id="b", downstream_task_id="d"),
        Dependency(upstream_task_id="c", downstream_task_id="d"),
    ]))
    plan.validate_dag()  # Should not raise


def test_dag_cycle_detection_simple():
    """Verify cycle detection: A → B → A."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Cycle")
    plan.add_task(_make_task("a", "plan-1", deps=[
        Dependency(upstream_task_id="b", downstream_task_id="a"),
    ]))
    plan.add_task(_make_task("b", "plan-1", deps=[
        Dependency(upstream_task_id="a", downstream_task_id="b"),
    ]))
    with pytest.raises(ValueError, match="cycle"):
        plan.validate_dag()


def test_dag_cycle_detection_three_way():
    """Verify cycle detection: A → B → C → A."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Three-way cycle")
    plan.add_task(_make_task("a", "plan-1", deps=[
        Dependency(upstream_task_id="c", downstream_task_id="a"),
    ]))
    plan.add_task(_make_task("b", "plan-1", deps=[
        Dependency(upstream_task_id="a", downstream_task_id="b"),
    ]))
    plan.add_task(_make_task("c", "plan-1", deps=[
        Dependency(upstream_task_id="b", downstream_task_id="c"),
    ]))
    with pytest.raises(ValueError, match="cycle"):
        plan.validate_dag()


def test_dag_missing_upstream_raises():
    """Verify dependency on nonexistent task raises ValueError."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Missing dep")
    plan.add_task(_make_task("a", "plan-1", deps=[
        Dependency(upstream_task_id="nonexistent", downstream_task_id="a"),
    ]))
    with pytest.raises(ValueError, match="unknown task 'nonexistent'"):
        plan.validate_dag()


def test_dag_validation_runs_on_activate():
    """Verify activate() rejects a plan with a cyclic DAG."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Activate with cycle")
    plan.add_task(_make_task("a", "plan-1", deps=[
        Dependency(upstream_task_id="b", downstream_task_id="a"),
    ]))
    plan.add_task(_make_task("b", "plan-1", deps=[
        Dependency(upstream_task_id="a", downstream_task_id="b"),
    ]))
    with pytest.raises(ValueError, match="cycle"):
        plan.activate()
    assert plan.status == PlanStatus.DRAFT  # Activation failed, still DRAFT


# ---------------------------------------------------------------------------
# PlanRevision
# ---------------------------------------------------------------------------

def test_plan_revision_recording():
    """Verify revision recording with sequential numbering."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))

    rev1 = plan.record_revision("rev-1", "Initial plan")
    assert rev1.revision_number == 1
    assert rev1.plan_id == "plan-1"
    assert rev1.reason == "Initial plan"
    assert rev1.task_ids == ["task-a"]
    assert rev1.created_at is not None

    plan.add_task(_make_task("task-b", "plan-1"))
    rev2 = plan.record_revision("rev-2", "Added task-b")
    assert rev2.revision_number == 2
    assert sorted(rev2.task_ids) == ["task-a", "task-b"]

    assert len(plan.revisions) == 2


def test_plan_revision_is_immutable():
    """Verify PlanRevision is frozen."""
    rev = PlanRevision(
        revision_id="rev-1",
        plan_id="plan-1",
        revision_number=1,
        reason="Test",
        task_ids=["task-a"],
    )
    with pytest.raises(AttributeError):
        rev.reason = "Changed"  # type: ignore[misc]


def test_plan_revision_preserves_snapshot():
    """Verify revision captures task IDs at recording time, not live."""
    plan = Plan(plan_id="plan-1", goal_id="goal-1", title="Test Plan")
    plan.add_task(_make_task("task-a", "plan-1"))
    rev1 = plan.record_revision("rev-1", "Snapshot with task-a only")

    plan.add_task(_make_task("task-b", "plan-1"))
    # rev1 should still only contain task-a
    assert rev1.task_ids == ["task-a"]
