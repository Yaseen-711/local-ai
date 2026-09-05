"""Repository protocols for orchestration domain persistence.

Defines the contract for persisting and loading Goal and Plan aggregate roots.
These interfaces are pure Python protocols and have zero dependencies on
SQLAlchemy or PostgreSQL.
"""

from typing import ContextManager, List, Optional, Protocol

from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan


class GoalRepository(Protocol):
    """Repository boundary for the Goal aggregate root."""

    def save(self, goal: Goal) -> None:
        """Atomically persist or update a Goal entity."""
        ...

    def get(self, goal_id: str) -> Optional[Goal]:
        """Fetch a Goal by its unique identifier."""
        ...

    def list_goals(self, limit: int = 100, offset: int = 0) -> List[Goal]:
        """List goals ordered by creation time descending."""
        ...


class PlanRepository(Protocol):
    """Repository boundary for the Plan aggregate root.

    Responsible for persisting and loading the entire Plan aggregate,
    including its tasks, DAG dependencies, execution attempts, and revisions.
    """

    def save(self, plan: Plan) -> None:
        """Atomically persist or update an entire Plan aggregate.

        Synchronizes the Plan, its tasks, attempts, and dependencies, and records
        new plan revisions in a single database transaction.
        """
        ...

    def get(self, plan_id: str) -> Optional[Plan]:
        """Load a complete Plan aggregate including all tasks, dependencies,
        attempts, results, and revisions.
        """
        ...

    def list_for_goal(self, goal_id: str) -> List[Plan]:
        """List all plans associated with a specific goal."""
        ...

    def get_historical_plan_revision(
        self, plan_id: str, revision_number: int
    ) -> Optional[Plan]:
        """Reconstitute a historical Plan revision using its immutable snapshot payload."""
        ...


class OrchestrationRepository(Protocol):
    """Unified boundary for orchestration domain persistence."""

    @property
    def goals(self) -> GoalRepository:
        """Access the Goal repository."""
        ...

    @property
    def plans(self) -> PlanRepository:
        """Access the Plan repository."""
        ...

    def transaction(self) -> ContextManager[None]:
        """Context manager providing an atomic transaction boundary."""
        ...
