"""Data structures and types for candidate plans and planning context."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan, PlanRevision
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import PlanStatus, TaskStatus


@dataclass
class CandidateTask:
    """Proposed task specification within a candidate plan.
    
    CandidateTasks are proposals that have not yet been validated or added to an active Plan.
    """
    task_id: str
    title: str
    capability_id: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_references: Dict[str, DataReference] = field(default_factory=dict)
    dependencies: List[Dependency] = field(default_factory=list)


@dataclass
class CandidatePlan:
    """Proposed plan produced by a Planner, prior to validation and activation.
    
    Planners propose CandidatePlans; they do NOT activate or execute them.

    Attributes:
        reused_tasks: Optional mapping of task_id → live domain Task for tasks
            being carried over from a prior plan with their full execution state
            (status, result, attempts, error, timestamps) intact.
            When to_plan() encounters a task_id present here it copies the
            domain Task directly instead of constructing a fresh PENDING Task.
    """
    plan_id: str
    goal_id: str
    title: str
    tasks: List[CandidateTask] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reused_tasks: Dict[str, Task] = field(default_factory=dict)
    prior_revisions: List[PlanRevision] = field(default_factory=list)

    def to_plan(self) -> Plan:
        """Convert this candidate plan to an unactivated domain Plan in DRAFT status.

        Reused tasks (present in ``reused_tasks``) are added to the plan with
        their original task_id, declarative specification, execution status,
        result, attempts, and error fully intact.  Only new tasks (not in
        ``reused_tasks``) are constructed fresh with PENDING status.
        Prior revisions from ``prior_revisions`` are preserved on the plan.
        """
        plan = Plan(
            plan_id=self.plan_id,
            goal_id=self.goal_id,
            title=self.title,
            status=PlanStatus.DRAFT,
            revisions=list(self.prior_revisions),
            initial_inputs=dict(self.metadata.get("inputs", {})),
        )
        for ct in self.tasks:
            # Consolidate dependencies: both task-level and plan-level targeting this task
            task_deps = list(ct.dependencies)
            for dep in self.dependencies:
                if dep.downstream_task_id == ct.task_id and dep not in task_deps:
                    task_deps.append(dep)

            if ct.task_id in self.reused_tasks:
                # Reuse the live domain Task — preserves all execution state.
                # We do NOT recreate a Task from the CandidateTask specification;
                # the prior Task's declarative spec is already immutable and correct.
                task = self.reused_tasks[ct.task_id]
            else:
                task = Task(
                    task_id=ct.task_id,
                    plan_id=self.plan_id,
                    title=ct.title,
                    capability_id=ct.capability_id,
                    description=ct.description,
                    parameters=dict(ct.parameters),
                    input_references=dict(ct.input_references),
                    dependencies=task_deps,
                )
            plan.add_task(task)

        return plan


@dataclass
class PlanningContext:
    """Context provided to Planners for initial planning or replanning."""
    goal: Goal
    prior_plan: Optional[Plan] = None
    prior_revision: Optional[PlanRevision] = None
    completed_tasks: Dict[str, TaskResult] = field(default_factory=dict)
    available_task_ids: Set[str] = field(default_factory=set)
    available_artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prior_plan:
            for tid, t in self.prior_plan.tasks.items():
                if t.status == TaskStatus.COMPLETED and t.result is not None:
                    self.completed_tasks.setdefault(tid, t.result)
                    self.available_task_ids.add(tid)
