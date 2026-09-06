"""Replanner handling work reuse and append-only PlanRevision generation."""

import asyncio
from typing import Dict, List, Optional, Set
import uuid

from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan, PlanRevision
from orchestration.domain.results import TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import TaskStatus
from orchestration.planning.base import Planner
from orchestration.planning.types import CandidatePlan, CandidateTask, PlanningContext


class Replanner:
    """Produces revised candidate plans while preserving completed work and revision history.
    
    Invariants enforced:
      1. Completed tasks from the prior plan are reused; their execution results are retained.
      2. Tasks requiring changes receive new task IDs (preserving declarative specification immutability).
      3. An append-only PlanRevision records the state transition history.
    """

    def __init__(self, planner: Planner) -> None:
        self._planner = planner

    def replan(
        self,
        goal: Goal,
        prior_plan: Plan,
        reason: str,
        preserve_completed: bool = True,
    ) -> CandidatePlan:
        """Produce a revised CandidatePlan building upon existing completed work."""
        completed_task_results: Dict[str, TaskResult] = {}
        completed_task_ids: Set[str] = set()

        if preserve_completed:
            for tid, t in prior_plan.tasks.items():
                if t.status == TaskStatus.COMPLETED and t.result is not None:
                    completed_task_results[tid] = t.result
                    completed_task_ids.add(tid)

        context = PlanningContext(
            goal=goal,
            prior_plan=prior_plan,
            prior_revision=prior_plan.revisions[-1] if prior_plan.revisions else None,
            completed_tasks=completed_task_results,
            available_task_ids=completed_task_ids,
            metadata={"replanning_reason": reason},
        )

        # Delegate proposal to underlying planner
        candidate = self._planner.plan(context)

        # Ensure completed tasks are preserved in candidate if requested.
        # Collect the live domain Task objects so to_plan() can restore their
        # full execution state (status, result, attempts, error, timestamps).
        existing_candidate_tids = {ct.task_id for ct in candidate.tasks}
        reused_candidate_tasks: List[CandidateTask] = []
        reused_domain_tasks: Dict[str, Task] = {}

        if preserve_completed:
            for cid in completed_task_ids:
                prior_t = prior_plan.tasks[cid]
                # Always populate reused_domain_tasks so to_plan() can restore
                # execution state even when the planner included the task_id in
                # its own proposal (e.g. as a dependency placeholder).
                reused_domain_tasks[cid] = prior_t

                if cid not in existing_candidate_tids:
                    reused_candidate_tasks.append(
                        CandidateTask(
                            task_id=prior_t.task_id,
                            title=prior_t.title,
                            capability_id=prior_t.capability_id,
                            description=prior_t.description,
                            parameters=dict(prior_t.parameters),
                            input_references=dict(prior_t.input_references),
                            dependencies=list(prior_t.dependencies),
                        )
                    )

        combined_tasks = reused_candidate_tasks + candidate.tasks

        return CandidatePlan(
            plan_id=prior_plan.plan_id,
            goal_id=goal.goal_id,
            title=candidate.title or prior_plan.title,
            tasks=combined_tasks,
            dependencies=candidate.dependencies,
            metadata={
                "revision_reason": reason,
                "reused_task_count": len(reused_candidate_tasks),
                "is_replan": True,
            },
            reused_tasks=reused_domain_tasks,
            prior_revisions=list(prior_plan.revisions),
        )

    async def replan_async(
        self,
        goal: Goal,
        prior_plan: Plan,
        reason: str,
        preserve_completed: bool = True,
    ) -> CandidatePlan:
        """Asynchronously produce a revised CandidatePlan."""
        return await asyncio.to_thread(self.replan, goal, prior_plan, reason, preserve_completed)
