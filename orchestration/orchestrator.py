"""Goal-Plan Orchestrator.

Coordinates Goal lifecycle with Plan execution via the PlanRunner execution boundary.
Decoupled from execution engines, planning algorithms, capability registries, and persistence.
"""

from typing import Optional

from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.execution.base import PlanRunner


class GoalOrchestrator:
    """Coordinates Goal lifecycle by binding Plans and delegating execution.

    Owns only Goal/Plan coordination and lifecycle interpretation:
    - Validates Goal and Plan identity and lifecycle states.
    - Binds Goal to Plan (goal.activate).
    - Delegates Plan execution to the PlanRunner execution boundary.
    - Synchronizes Goal status with Plan outcome (COMPLETED, FAILED, CANCELLED).
    - Cancels Goal and associated Plan/Tasks on demand.
    """

    def __init__(self, runner: PlanRunner) -> None:
        """Initialize orchestrator with an execution boundary runner.

        Args:
            runner: Object satisfying the PlanRunner execution boundary protocol.
        """
        self._runner = runner

    @property
    def runner(self) -> PlanRunner:
        """Underlying execution boundary runner."""
        return self._runner

    def execute_goal(self, goal: Goal, plan: Plan) -> Goal:
        """Bind a Plan to a Goal, execute the plan, and synchronize Goal status.

        Args:
            goal: The Goal to execute (must be in PENDING status).
            plan: The Plan strategy (must be in DRAFT or ACTIVE status,
                and its goal_id must match goal.goal_id).

        Returns:
            The Goal updated to its terminal state (COMPLETED, FAILED, or CANCELLED).

        Raises:
            ValueError: If goal is not in PENDING status, plan is not in
                DRAFT or ACTIVE status, or plan.goal_id does not match goal.goal_id.
        """
        if goal.status != GoalStatus.PENDING:
            raise ValueError(
                f"Cannot execute goal '{goal.goal_id}': status is '{goal.status.value}', "
                f"expected 'pending'."
            )

        if plan.goal_id != goal.goal_id:
            raise ValueError(
                f"Cannot execute goal '{goal.goal_id}': plan '{plan.plan_id}' belongs to "
                f"goal '{plan.goal_id}', expected '{goal.goal_id}'."
            )

        if plan.status not in (PlanStatus.DRAFT, PlanStatus.ACTIVE):
            raise ValueError(
                f"Cannot execute plan '{plan.plan_id}': status is '{plan.status.value}', "
                f"expected 'draft' or 'active'."
            )

        # Bind goal to plan
        goal.activate(plan.plan_id)

        # Delegate execution to runner within safety boundary
        try:
            executed_plan = self._runner.run(plan)
        except Exception:
            # Prevent goal from remaining stranded in ACTIVE on unhandled runner failures
            if goal.status == GoalStatus.ACTIVE:
                goal.mark_failed()
            raise

        # Synchronize Goal status with Plan outcome
        if executed_plan.status == PlanStatus.COMPLETED:
            goal.mark_completed()
        elif executed_plan.status == PlanStatus.FAILED:
            goal.mark_failed()
        elif executed_plan.status == PlanStatus.CANCELLED:
            goal.cancel()

        return goal

    def cancel_goal(self, goal: Goal, plan: Optional[Plan] = None) -> None:
        """Cancel a Goal and optionally its associated Plan and Tasks.

        Args:
            goal: The Goal to cancel.
            plan: Optional active Plan associated with the Goal.

        Raises:
            ValueError: If plan belongs to a different goal, or if goal
                is already in COMPLETED status.
        """
        if plan is not None:
            if plan.goal_id != goal.goal_id:
                raise ValueError(
                    f"Cannot cancel goal '{goal.goal_id}': plan '{plan.plan_id}' belongs to "
                    f"goal '{plan.goal_id}', expected '{goal.goal_id}'."
                )

            # Cancel uncompleted tasks first
            for task in plan.tasks.values():
                if task.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                    task.cancel()

            if plan.status != PlanStatus.CANCELLED:
                plan.cancel()

        if goal.status != GoalStatus.CANCELLED:
            goal.cancel()
