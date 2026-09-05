"""Plan and PlanRevision — strategy and revision history for goals.

A Plan organizes a set of Tasks into a directed acyclic graph (DAG) and
manages their collective lifecycle. PlanRevision is an immutable snapshot
recording the plan's task composition at a point in time, preserving
planning history without full event sourcing or persistence.

Plan lifecycle:
    DRAFT     → ACTIVE      (plan activated for execution)
    ACTIVE    → COMPLETED   (all tasks completed)
    ACTIVE    → FAILED      (a required task failed irrecoverably)
    Any       → CANCELLED   (external cancellation)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from orchestration.domain.types import PlanStatus, TaskStatus
from orchestration.domain.tasks import Task


@dataclass(frozen=True)
class PlanRevision:
    """Immutable snapshot of a plan's task composition at a point in time.

    Preserves planning history so that future plan changes do not
    overwrite prior states. Does not implement persistence, event
    sourcing, or diff engines.

    Attributes:
        revision_id: Unique identifier for this revision.
        plan_id: ID of the plan this revision belongs to.
        revision_number: Sequential revision number (1-based).
        reason: Human-readable reason for this revision.
        task_ids: Task IDs present in the plan at revision time.
        created_at: When this revision was recorded.
    """
    revision_id: str
    plan_id: str
    revision_number: int
    reason: str
    task_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Plan:
    """Strategy to achieve a Goal, managing a DAG of tasks.

    Attributes:
        plan_id: Unique identifier for this plan.
        goal_id: ID of the goal this plan serves.
        title: Human-readable plan title.
        status: Current lifecycle state.
        tasks: Task entities keyed by task_id.
        revisions: Ordered revision history.
        created_at: When this plan was created.
        completed_at: When this plan reached a terminal state.
    """
    plan_id: str
    goal_id: str
    title: str
    status: PlanStatus = PlanStatus.DRAFT
    tasks: Dict[str, Task] = field(default_factory=dict)
    revisions: List[PlanRevision] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def add_task(self, task: Task) -> None:
        """Add a task to this plan.

        Args:
            task: Task entity to add. Its plan_id must match this plan.

        Raises:
            ValueError: If the plan is not in DRAFT or ACTIVE status,
                or if the task's plan_id does not match.
        """
        if self.status not in (PlanStatus.DRAFT, PlanStatus.ACTIVE):
            raise ValueError(
                f"Cannot add task to plan '{self.plan_id}': "
                f"plan status is '{self.status.value}'."
            )
        if task.plan_id != self.plan_id:
            raise ValueError(
                f"Task '{task.task_id}' has plan_id '{task.plan_id}', "
                f"expected '{self.plan_id}'."
            )
        if task.task_id in self.tasks:
            raise ValueError(
                f"Task '{task.task_id}' already exists in plan '{self.plan_id}'."
            )
        self.tasks[task.task_id] = task

    def activate(self) -> None:
        """Transition plan from DRAFT to ACTIVE.

        Validates the task DAG before activation.

        Raises:
            ValueError: If the plan is not in DRAFT status, has no tasks,
                or contains dependency cycles.
        """
        if self.status != PlanStatus.DRAFT:
            raise ValueError(
                f"Cannot activate plan '{self.plan_id}': "
                f"status is '{self.status.value}', expected 'draft'."
            )
        if not self.tasks:
            raise ValueError(
                f"Cannot activate plan '{self.plan_id}': plan has no tasks."
            )
        self.validate_dag()
        self.status = PlanStatus.ACTIVE

    def record_revision(self, revision_id: str, reason: str) -> PlanRevision:
        """Record a snapshot of the current plan state.

        Args:
            revision_id: Unique identifier for the revision.
            reason: Human-readable reason for the revision.

        Returns:
            The newly created PlanRevision.
        """
        rev_num = len(self.revisions) + 1
        rev = PlanRevision(
            revision_id=revision_id,
            plan_id=self.plan_id,
            revision_number=rev_num,
            reason=reason,
            task_ids=list(self.tasks.keys()),
        )
        self.revisions.append(rev)
        return rev

    def validate_dag(self) -> None:
        """Validate that task dependencies form a valid DAG.

        Checks:
        1. All dependency upstream task IDs reference tasks in this plan.
        2. No dependency cycles exist.

        Raises:
            ValueError: If a dependency references an unknown task or
                a cycle is detected.
        """
        # Verify all dependency targets exist in this plan
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep.upstream_task_id not in self.tasks:
                    raise ValueError(
                        f"Task '{task.task_id}' depends on unknown task "
                        f"'{dep.upstream_task_id}' (not in plan '{self.plan_id}')."
                    )

        # DFS cycle detection
        # States: 0 = unvisited, 1 = visiting (in current DFS path), 2 = visited
        visited: Dict[str, int] = {tid: 0 for tid in self.tasks}

        def _dfs(task_id: str) -> None:
            visited[task_id] = 1
            task = self.tasks[task_id]
            for dep in task.dependencies:
                upstream_id = dep.upstream_task_id
                if visited[upstream_id] == 1:
                    raise ValueError(
                        f"Dependency cycle detected: task '{task_id}' "
                        f"has a circular dependency involving '{upstream_id}'."
                    )
                if visited[upstream_id] == 0:
                    _dfs(upstream_id)
            visited[task_id] = 2

        for task_id in self.tasks:
            if visited[task_id] == 0:
                _dfs(task_id)

    def mark_completed(self) -> None:
        """Mark the plan as COMPLETED.

        Raises:
            ValueError: If the plan is not ACTIVE.
        """
        if self.status != PlanStatus.ACTIVE:
            raise ValueError(
                f"Cannot complete plan '{self.plan_id}': "
                f"status is '{self.status.value}', expected 'active'."
            )
        self.status = PlanStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self) -> None:
        """Mark the plan as FAILED.

        Raises:
            ValueError: If the plan is not ACTIVE.
        """
        if self.status != PlanStatus.ACTIVE:
            raise ValueError(
                f"Cannot fail plan '{self.plan_id}': "
                f"status is '{self.status.value}', expected 'active'."
            )
        self.status = PlanStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)

    def cancel(self) -> None:
        """Cancel the plan.

        Raises:
            ValueError: If the plan is already completed.
        """
        if self.status == PlanStatus.CANCELLED:
            return  # Idempotent
        if self.status == PlanStatus.COMPLETED:
            raise ValueError(
                f"Cannot cancel plan '{self.plan_id}': already completed."
            )
        self.status = PlanStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
