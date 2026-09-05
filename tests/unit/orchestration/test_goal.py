"""Unit tests for Goal lifecycle and invariants."""

import pytest

from orchestration.domain.goals import Goal
from orchestration.domain.types import GoalStatus


def test_goal_creation_defaults():
    """Verify Goal is created with PENDING status and sensible defaults."""
    goal = Goal(goal_id="goal-1", description="Analyze quarterly report")
    assert goal.goal_id == "goal-1"
    assert goal.description == "Analyze quarterly report"
    assert goal.status == GoalStatus.PENDING
    assert goal.active_plan_id is None
    assert goal.context == {}
    assert goal.completed_at is None
    assert goal.created_at is not None


def test_goal_activate():
    """Verify PENDING → ACTIVE transition with plan association."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    goal.activate(plan_id="plan-1")
    assert goal.status == GoalStatus.ACTIVE
    assert goal.active_plan_id == "plan-1"


def test_goal_activate_not_pending_raises():
    """Verify activation from non-PENDING state raises ValueError."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    goal.activate(plan_id="plan-1")
    with pytest.raises(ValueError, match="expected 'pending'"):
        goal.activate(plan_id="plan-2")


def test_goal_mark_completed():
    """Verify ACTIVE → COMPLETED transition."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    goal.activate(plan_id="plan-1")
    goal.mark_completed()
    assert goal.status == GoalStatus.COMPLETED
    assert goal.completed_at is not None


def test_goal_mark_completed_not_active_raises():
    """Verify completion from non-ACTIVE state raises ValueError."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    with pytest.raises(ValueError, match="expected 'active'"):
        goal.mark_completed()


def test_goal_mark_failed():
    """Verify ACTIVE → FAILED transition."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    goal.activate(plan_id="plan-1")
    goal.mark_failed()
    assert goal.status == GoalStatus.FAILED
    assert goal.completed_at is not None


def test_goal_mark_failed_not_active_raises():
    """Verify failure from non-ACTIVE state raises ValueError."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    with pytest.raises(ValueError, match="expected 'active'"):
        goal.mark_failed()


def test_goal_cancel_from_pending():
    """Verify cancellation from PENDING."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    goal.cancel()
    assert goal.status == GoalStatus.CANCELLED
    assert goal.completed_at is not None


def test_goal_cancel_from_active():
    """Verify cancellation from ACTIVE."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    goal.activate(plan_id="plan-1")
    goal.cancel()
    assert goal.status == GoalStatus.CANCELLED


def test_goal_cancel_completed_raises():
    """Verify cancellation of COMPLETED goal raises ValueError."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    goal.activate(plan_id="plan-1")
    goal.mark_completed()
    with pytest.raises(ValueError, match="already completed"):
        goal.cancel()


def test_goal_cancel_idempotent():
    """Verify cancelling an already-cancelled goal is a no-op."""
    goal = Goal(goal_id="goal-1", description="Test goal")
    goal.cancel()
    goal.cancel()  # Should not raise
    assert goal.status == GoalStatus.CANCELLED


def test_goal_context_metadata():
    """Verify arbitrary context metadata is preserved."""
    goal = Goal(
        goal_id="goal-1",
        description="Test goal",
        context={"priority": "high", "requester": "user-42"},
    )
    assert goal.context["priority"] == "high"
    assert goal.context["requester"] == "user-42"
