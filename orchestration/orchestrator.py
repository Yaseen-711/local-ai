"""Goal-Plan Orchestrator.

Coordinates Goal lifecycle with Plan execution via the PlanRunner execution boundary.
Decoupled from execution engines, planning algorithms, capability registries, and persistence.
"""

import asyncio
from typing import Dict, Optional

from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.execution.base import (
    ExecutionHandle,
    PlanExecutionResult,
    PlanRunner,
)


class GoalOrchestrator:
    """Coordinates Goal lifecycle by binding Plans and delegating execution.

    Owns only Goal/Plan coordination and lifecycle interpretation:
    - Validates Goal and Plan identity and lifecycle states.
    - Binds Goal to Plan (goal.activate).
    - Delegates Plan execution to the PlanRunner execution boundary.
    - Applies execution facts to Plan and Tasks, preserving domain invariants.
    - Synchronizes Goal status with Plan outcome (COMPLETED, FAILED, CANCELLED).
    - Cancels Goal, active executions via PlanRunner, and associated Plan/Tasks on demand.
    """

    def __init__(self, runner: PlanRunner) -> None:
        """Initialize orchestrator with an execution boundary runner.

        Args:
            runner: Object satisfying the PlanRunner execution boundary protocol.
        """
        self._runner = runner
        self._active_handles: Dict[str, ExecutionHandle] = {}

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
        self._validate_goal_and_plan(goal, plan)

        # Bind goal to plan
        goal.activate(plan.plan_id)

        # Delegate execution to runner within safety boundary
        try:
            if hasattr(self._runner, "start") and hasattr(self._runner, "wait"):
                handle = self._runner.start(plan)
                self._active_handles[goal.goal_id] = handle
                try:
                    result = self._runner.wait(handle)
                finally:
                    self._active_handles.pop(goal.goal_id, None)

                if isinstance(result, PlanExecutionResult):
                    self._apply_execution_result(plan, result)
            else:
                self._runner.run(plan)
        except Exception:
            # Prevent goal from remaining stranded in ACTIVE on unhandled runner failures
            if goal.status == GoalStatus.ACTIVE:
                goal.mark_failed()
            raise

        self._sync_goal_with_plan(goal, plan)
        return goal

    async def execute_goal_async(self, goal: Goal, plan: Plan) -> Goal:
        """Asynchronously bind a Plan to a Goal, execute the plan, and synchronize Goal status.

        Supports native asynchronous execution runners (e.g. TemporalPlanRunner)
        without blocking the caller's asyncio event loop.

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
        self._validate_goal_and_plan(goal, plan)

        # Bind goal to plan
        goal.activate(plan.plan_id)

        try:
            if hasattr(self._runner, "start_async") and hasattr(self._runner, "wait_async"):
                handle = await self._runner.start_async(plan)
                self._active_handles[goal.goal_id] = handle
                try:
                    result = await self._runner.wait_async(handle)
                finally:
                    self._active_handles.pop(goal.goal_id, None)

                if isinstance(result, PlanExecutionResult):
                    self._apply_execution_result(plan, result)
            elif hasattr(self._runner, "start") and hasattr(self._runner, "wait"):
                handle = await asyncio.to_thread(self._runner.start, plan)
                self._active_handles[goal.goal_id] = handle
                try:
                    result = await asyncio.to_thread(self._runner.wait, handle)
                finally:
                    self._active_handles.pop(goal.goal_id, None)

                if isinstance(result, PlanExecutionResult):
                    self._apply_execution_result(plan, result)
            else:
                await asyncio.to_thread(self._runner.run, plan)
        except Exception:
            if goal.status == GoalStatus.ACTIVE:
                goal.mark_failed()
            raise

        self._sync_goal_with_plan(goal, plan)
        return goal

    def cancel_goal(self, goal: Goal, plan: Optional[Plan] = None) -> None:
        """Cancel a Goal, cancel active runner execution, and cancel Plan/Tasks.

        Args:
            goal: The Goal to cancel.
            plan: Optional active Plan associated with the Goal.

        Raises:
            ValueError: If plan belongs to a different goal, or if goal
                is already in COMPLETED status.
        """
        if goal.status == GoalStatus.COMPLETED:
            raise ValueError(
                f"Cannot cancel goal '{goal.goal_id}': already completed."
            )

        if plan is not None:
            if plan.goal_id != goal.goal_id:
                raise ValueError(
                    f"Cannot cancel goal '{goal.goal_id}': plan '{plan.plan_id}' belongs to "
                    f"goal '{plan.goal_id}', expected '{goal.goal_id}'."
                )

        # Delegate cancellation to running execution backend if active
        active_handle = self._active_handles.pop(goal.goal_id, None)
        if active_handle is not None:
            self._runner.cancel(active_handle)

        if plan is not None:
            for task in plan.tasks.values():
                if task.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                    task.cancel()

            if plan.status != PlanStatus.CANCELLED:
                plan.cancel()

        if goal.status != GoalStatus.CANCELLED:
            goal.cancel()

    async def cancel_goal_async(self, goal: Goal, plan: Optional[Plan] = None) -> None:
        """Asynchronously cancel a Goal, propagating cancellation to the runner.

        Args:
            goal: The Goal to cancel.
            plan: Optional active Plan associated with the Goal.

        Raises:
            ValueError: If plan belongs to a different goal, or if goal
                is already in COMPLETED status.
        """
        if goal.status == GoalStatus.COMPLETED:
            raise ValueError(
                f"Cannot cancel goal '{goal.goal_id}': already completed."
            )

        if plan is not None:
            if plan.goal_id != goal.goal_id:
                raise ValueError(
                    f"Cannot cancel goal '{goal.goal_id}': plan '{plan.plan_id}' belongs to "
                    f"goal '{plan.goal_id}', expected '{goal.goal_id}'."
                )

        active_handle = self._active_handles.pop(goal.goal_id, None)
        if active_handle is not None:
            if hasattr(self._runner, "cancel_async"):
                await self._runner.cancel_async(active_handle)
            else:
                await asyncio.to_thread(self._runner.cancel, active_handle)

        if plan is not None:
            for task in plan.tasks.values():
                if task.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                    task.cancel()

            if plan.status != PlanStatus.CANCELLED:
                plan.cancel()

        if goal.status != GoalStatus.CANCELLED:
            goal.cancel()

    def _validate_goal_and_plan(self, goal: Goal, plan: Plan) -> None:
        """Validate Goal and Plan eligibility for execution."""
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

    def _sync_goal_with_plan(self, goal: Goal, plan: Plan) -> None:
        """Synchronize Goal lifecycle status with the Plan outcome."""
        if plan.status == PlanStatus.COMPLETED:
            goal.mark_completed()
        elif plan.status == PlanStatus.FAILED:
            goal.mark_failed()
        elif plan.status == PlanStatus.CANCELLED:
            goal.cancel()

    def _apply_execution_result(self, plan: Plan, result: PlanExecutionResult) -> None:
        """Apply terminal execution facts to domain Plan and Tasks adhering to domain invariants."""
        # Drive task readiness and attempt transitions in dependency order
        changed = True
        while changed:
            changed = False
            current_statuses = {t_id: t.status for t_id, t in plan.tasks.items()}
            for t_id, task in plan.tasks.items():
                if task.status == TaskStatus.PENDING:
                    task.update_readiness(current_statuses)
                    if task.status != TaskStatus.PENDING:
                        changed = True

                if task.status == TaskStatus.READY:
                    attempt_id = result.task_attempts.get(t_id) or f"att-{t_id}-{len(task.attempts) + 1}"
                    if t_id in result.task_results:
                        task.start_attempt(attempt_id)
                        task.complete_attempt(attempt_id, result.task_results[t_id])
                        changed = True
                    elif t_id in result.task_errors:
                        task.start_attempt(attempt_id)
                        task.fail_attempt(attempt_id, result.task_errors[t_id])
                        changed = True

        # Handle cancellations or unstarted tasks if plan was cancelled
        for task in plan.tasks.values():
            if result.status == PlanStatus.CANCELLED and task.status not in (
                TaskStatus.COMPLETED,
                TaskStatus.CANCELLED,
            ):
                task.cancel()

        # Update plan status
        if result.status == PlanStatus.COMPLETED and plan.status != PlanStatus.COMPLETED:
            if plan.status == PlanStatus.DRAFT:
                plan.activate()
            plan.mark_completed()
        elif result.status == PlanStatus.FAILED and plan.status != PlanStatus.FAILED:
            if plan.status == PlanStatus.DRAFT:
                plan.activate()
            plan.mark_failed()
        elif result.status == PlanStatus.CANCELLED and plan.status != PlanStatus.CANCELLED:
            if plan.status == PlanStatus.DRAFT:
                plan.activate()
            plan.cancel()
