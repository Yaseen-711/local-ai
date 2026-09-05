"""Goal entity — high-level objective and intent container.

A Goal represents the user's or system's high-level objective. It
owns a reference to its active plan but does not manage plan internals.

Lifecycle:
    PENDING   → ACTIVE      (work begins)
    ACTIVE    → COMPLETED   (objective achieved)
    ACTIVE    → FAILED      (objective cannot be achieved)
    Any       → CANCELLED   (abandoned)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orchestration.domain.types import GoalStatus


@dataclass
class Goal:
    """High-level objective and intent container.

    Attributes:
        goal_id: Unique identifier for this goal.
        description: Human-readable description of the objective.
        status: Current lifecycle state.
        context: Arbitrary contextual metadata for the goal
            (e.g. user-provided constraints, preferences, scope).
        active_plan_id: ID of the currently active plan, if any.
        created_at: When this goal was created.
        completed_at: When this goal reached a terminal state.
    """
    goal_id: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    context: Dict[str, Any] = field(default_factory=dict)
    active_plan_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def activate(self, plan_id: str) -> None:
        """Activate the goal with a specific plan.

        Args:
            plan_id: ID of the plan to associate.

        Raises:
            ValueError: If the goal is not in PENDING status.
        """
        if self.status != GoalStatus.PENDING:
            raise ValueError(
                f"Cannot activate goal '{self.goal_id}': "
                f"status is '{self.status.value}', expected 'pending'."
            )
        self.active_plan_id = plan_id
        self.status = GoalStatus.ACTIVE

    def mark_completed(self) -> None:
        """Mark the goal as COMPLETED.

        Raises:
            ValueError: If the goal is not ACTIVE.
        """
        if self.status != GoalStatus.ACTIVE:
            raise ValueError(
                f"Cannot complete goal '{self.goal_id}': "
                f"status is '{self.status.value}', expected 'active'."
            )
        self.status = GoalStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self) -> None:
        """Mark the goal as FAILED.

        Raises:
            ValueError: If the goal is not ACTIVE.
        """
        if self.status != GoalStatus.ACTIVE:
            raise ValueError(
                f"Cannot fail goal '{self.goal_id}': "
                f"status is '{self.status.value}', expected 'active'."
            )
        self.status = GoalStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)

    def cancel(self) -> None:
        """Cancel the goal.

        Raises:
            ValueError: If the goal is already completed.
        """
        if self.status == GoalStatus.CANCELLED:
            return  # Idempotent
        if self.status == GoalStatus.COMPLETED:
            raise ValueError(
                f"Cannot cancel goal '{self.goal_id}': already completed."
            )
        self.status = GoalStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
